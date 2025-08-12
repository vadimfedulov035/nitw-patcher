#!/usr/bin/env python

import os
import re
import json
import shutil
import curses
import pyfiglet
from collections import defaultdict
from typing import List, Tuple, Optional, Callable, Iterator

# --- CONFIGURATION ---
ORIGINAL_DIR = "original"
PATCHED_DIR = "patched"
EXTRACTED_DIR = "extracted"
ADDRESSES_FILE = os.path.join(EXTRACTED_DIR, "addresses.txt")
NAMES_FILE = os.path.join(EXTRACTED_DIR, "names.txt")
ORIGINAL_PHRASES_FILE = os.path.join(EXTRACTED_DIR, "original.txt")
TRANSLATED_PHRASES_FILE = os.path.join(EXTRACTED_DIR, "translated.txt")
VALIDATION_RULES_FILE = "validation_rules.json"
SPECIAL_PHRASE_PATTERN = re.compile(r"\[\[(?:\{.*?\})?(?P<PHRASE>.*?)(?:\{.*?\})?\|(?:.*?)\]\]\s*#line")
BASIC_PHRASE_PATTERN = re.compile(r"\s*(?:(?P<NAME>\w*?)(?:\:))?(?:->)?\s*(?:\{.*?\})*(?:\[.*?\])*\s*(?P<PHRASE>.*?)(?:\{.*?\})?(?:\[\/.*?\])?\s*#line")
PUNCTUATION_MARKS = ('.', '?', '!', '...', ',')
DEFAULT_NAME = "Not Stated"

# --- STYLE AND TUI CONFIGURATION ---
FIGLET_FONT = "starwars"
PROGRESS_BAR_FILLED = '█'
PROGRESS_BAR_EMPTY = '░'
CONFIRM_YES = "[ Yes ]"
CONFIRM_NO = "[ No ]"
COLOR_PAIR_SELECTED = 1
COLOR_PAIR_CANCEL = 2


# --- HELPERS ---

def _find_first_letter(line: str) -> Tuple[Optional[str], int]:
    """Finds the first alphabetic character and its index.

    Args:
        line (str): The string to search.

    Returns:
        tuple: A tuple containing the character (or None) and its index (-1 if not found).
    """
    for i, char in enumerate(line):
        if char.isalpha():
            return char, i
    return None, -1


def _has_word(line: str, word: str) -> bool:
    """Checks for a whole word in a line, case-insensitively.

    Args:
        line (str): The string to search within.
        word (str): The word to search for.

    Returns:
        bool: True if the whole word is found, False otherwise.
    """
    return bool(re.search(r'\b' + re.escape(word) + r'\b', line, re.IGNORECASE))


def _read_file_lines(filepath: str) -> Optional[List[str]]:
    """Reads all lines from a file into a list if it exists.

    Args:
        filepath (str): The path to the file.

    Returns:
        list or None: A list of stripped lines, or None if the file doesn't exist.
    """
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


def _check_files_exist(*filepaths: str) -> Optional[str]:
    """Checks if all given filepaths exist, returning an error message if not.

    Args:
        *filepaths (str): A variable number of file paths to check.

    Returns:
        str or None: An error message if a file is missing, otherwise None.
    """
    for fp in filepaths:
        if not os.path.exists(fp):
            return f"Error: Required file or directory '{fp}' not found."
    return None


# --- CORE ACTIONS ---

def extract(on_progress: Callable = None) -> Iterator:
    """Extracts phrases, names, and addresses from original game files.

    Args:
        on_progress (Callable, optional): A callback function to report progress.

    Yields:
        tuple or str: A status message or a tuple with findings for the UI controller.
    """
    error = _check_files_exist(ORIGINAL_DIR)
    if error: yield error; return
    os.makedirs(EXTRACTED_DIR, exist_ok=True)

    addresses, names, phrases = [], [], []
    original_files = os.listdir(ORIGINAL_DIR)

    for i, filename in enumerate(original_files, 1):
        if on_progress: on_progress(i, len(original_files), f"Extracting: {filename}")
        src_path = os.path.join(ORIGINAL_DIR, filename)
        with open(src_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                match = SPECIAL_PHRASE_PATTERN.search(line) or BASIC_PHRASE_PATTERN.search(line)
                if not match: continue
                name = match.groupdict().get("NAME") or DEFAULT_NAME
                start, end = match.span("PHRASE")
                phrase = line[start:end]
                addresses.append(f"{filename}:{line_num}:{start}:{end}")
                names.append(name)
                phrases.append(phrase)

    yield "write_files", addresses, names, phrases
    yield "Extraction complete. Press any key."


def normalize() -> Iterator:
    """Analyzes translated file for normalization and yields findings.

    Yields:
        str or tuple: A status message or a tuple with findings for the UI controller.
    """
    error = _check_files_exist(ORIGINAL_PHRASES_FILE, TRANSLATED_PHRASES_FILE)
    if error: yield error; return

    original_lines = _read_file_lines(ORIGINAL_PHRASES_FILE)
    translated_lines = _read_file_lines(TRANSLATED_PHRASES_FILE)

    if len(original_lines) != len(translated_lines):
        yield "Error: File line counts mismatch."; return

    normalized_lines, changes_count = [], 0
    for orig, trans in zip(original_lines, translated_lines):
        new_trans = trans
        orig_char, _ = _find_first_letter(orig)
        trans_char, trans_idx = _find_first_letter(new_trans)
        if orig_char and trans_char and orig_char.isupper() != trans_char.isupper():
            line_list = list(new_trans)
            line_list[trans_idx] = line_list[trans_idx].upper() if orig_char.isupper() else line_list[trans_idx].lower()
            new_trans = "".join(line_list)
        orig_punct = next((p for p in PUNCTUATION_MARKS if orig.endswith(p)), None)
        trans_punct = next((p for p in PUNCTUATION_MARKS if new_trans.endswith(p)), None)
        if orig_punct != trans_punct:
            if trans_punct: new_trans = new_trans[:-len(trans_punct)]
            if orig_punct: new_trans += orig_punct
        if new_trans != trans: changes_count += 1
        normalized_lines.append(new_trans)

    if changes_count == 0:
        yield "Normalization not needed. All lines conform."; return

    yield "confirm_normalize", changes_count, normalized_lines


def validate(on_progress: Callable = None) -> Iterator:
    """Validates translation against keyword rules from a JSON file.

    Args:
        on_progress (Callable, optional): A callback function to report progress.
    
    Yields:
        str or tuple: A status message or a tuple with findings for the UI controller.
    """
    error = _check_files_exist(VALIDATION_RULES_FILE, ORIGINAL_PHRASES_FILE, TRANSLATED_PHRASES_FILE)
    if error: yield error; return

    try:
        with open(VALIDATION_RULES_FILE, 'r', encoding='utf-8') as f:
            rules = json.load(f).get("rules", [])
    except json.JSONDecodeError:
        yield f"Error: Could not parse '{VALIDATION_RULES_FILE}'."; return

    original_lines = _read_file_lines(ORIGINAL_PHRASES_FILE)
    translated_lines = _read_file_lines(TRANSLATED_PHRASES_FILE)

    errors = []
    total_lines = len(original_lines)
    for i, (orig_line, trans_line) in enumerate(zip(original_lines, translated_lines)):
        if on_progress: on_progress(i + 1, total_lines, "Validating lines...")
        for rule in rules:
            en, eo_values = rule.get("en"), rule.get("eo")
            if not en or not eo_values: continue
            if not isinstance(eo_values, list): eo_values = [eo_values]
            if _has_word(orig_line, en) and not any(_has_word(trans_line, eo) for eo in eo_values):
                errors.append(f"L{i+1:04d}: Found '{en}' but missing one of: {eo_values}")
    
    if not errors:
        yield "Validation passed! All keyword rules are met."
    else:
        yield "confirm_view_errors", errors


def patch(on_progress: Callable = None) -> Iterator[str]:
    """Applies translated phrases to create patched game files.

    Args:
        on_progress (Callable, optional): A callback function to report progress.
        
    Yields:
        str: A status message indicating completion or an error.
    """
    error = _check_files_exist(TRANSLATED_PHRASES_FILE, ADDRESSES_FILE, ORIGINAL_DIR)
    if error: yield error; return

    translations = _read_file_lines(TRANSLATED_PHRASES_FILE)
    addresses = _read_file_lines(ADDRESSES_FILE)
    if len(translations) != len(addresses):
        yield "Error: Mismatch between translation and address files."; return

    file_map = defaultdict(list)
    for i, addr in enumerate(addresses):
        try:
            filename, line, start, end = addr.split(':', 3)
            file_map[filename].append((i, int(line), int(start), int(end)))
        except ValueError: continue

    if os.path.exists(PATCHED_DIR): shutil.rmtree(PATCHED_DIR)
    os.makedirs(PATCHED_DIR)

    for i, filename in enumerate(file_map, 1):
        if on_progress: on_progress(i, len(file_map), f"Patching: {filename}")
        src_path = os.path.join(ORIGINAL_DIR, filename)
        if not os.path.exists(src_path): continue
        
        with open(src_path, "r", encoding="utf-8") as src:
            lines = src.read().splitlines()
        for idx, line_num, start, end in sorted(file_map[filename], key=lambda x: x[1], reverse=True):
            if line_num - 1 < len(lines):
                lines[line_num-1] = lines[line_num-1][:start] + translations[idx] + lines[line_num-1][end:]
        with open(os.path.join(PATCHED_DIR, filename), "w", encoding="utf-8") as dst:
            dst.write("\n".join(lines))
    yield "Patching complete! Press any key."


# --- TUI ---

def view_errors(win: curses.window, title: str, lines: List[str]):
    """A scrollable text viewer for lists of strings.

    Args:
        win (curses.window): The curses window to draw in.
        title (str): The title to display at the top.
        lines (list): A list of strings to display.
    """
    h, w = win.getmaxyx()
    scroll_pos = 0
    while True:
        win.clear(); win.refresh()
        win.addstr(0, 2, f"{title} ({len(lines)} entries)", curses.A_BOLD)
        win.addstr(h - 1, 2, "UP/DOWN to scroll. 'q' to exit.")
        for i, line in enumerate(lines[scroll_pos : scroll_pos + h - 2]):
            win.addstr(i + 1, 2, line[:w-3])
        key = win.getch()
        if key == curses.KEY_UP and scroll_pos > 0: scroll_pos -= 1
        elif key == curses.KEY_DOWN and scroll_pos < len(lines) - (h - 2): scroll_pos += 1
        elif key == ord('q'): break


def show_progress(win: curses.window, current: int, total: int, message: str):
    """Displays a progress bar and a status message.

    Args:
        win (curses.window): The curses window to draw in.
        current (int): The current progress count.
        total (int): The total number for 100% progress.
        message (str): The message to display below the bar.
    """
    h, w = win.getmaxyx()
    win.clear()
    bar_width = min(40, w - 12)
    percent = current / total if total > 0 else 0
    filled_len = int(bar_width * percent)
    bar_display = f"[{PROGRESS_BAR_FILLED * filled_len}{PROGRESS_BAR_EMPTY * (bar_width - filled_len)}] {percent:.0%}"
    win.addstr(h // 2 - 1, w // 2 - len(bar_display) // 2, bar_display)
    win.addstr(h // 2 + 1, w // 2 - len(message) // 2, message)
    win.refresh()


def confirm(win: curses.window, prompt: str) -> bool:
    """Displays a Yes/No confirmation dialog.

    Args:
        win (curses.window): The curses window to draw in.
        prompt (str): The question to ask the user.

    Returns:
        bool: True if "Yes" is selected, False otherwise.
    """
    options, selection = [CONFIRM_YES, CONFIRM_NO], 0
    while True:
        h, w = win.getmaxyx(); win.clear()
        win.addstr(h // 2 - 2, w // 2 - len(prompt) // 2, prompt)
        yes_x = w // 2 - len("".join(options)) // 2
        no_x = yes_x + len(options[0]) + 2
        win.attron(curses.color_pair(COLOR_PAIR_SELECTED) if selection == 0 else curses.A_NORMAL)
        win.addstr(h // 2, yes_x, options[0])
        win.attroff(curses.color_pair(COLOR_PAIR_SELECTED))
        win.attron(curses.color_pair(COLOR_PAIR_CANCEL) if selection == 1 else curses.A_NORMAL)
        win.addstr(h // 2, no_x, options[1])
        win.attroff(curses.color_pair(COLOR_PAIR_CANCEL))
        win.refresh()
        key = win.getch()
        if key in [curses.KEY_LEFT, curses.KEY_RIGHT]: selection = 1 - selection
        elif key in [curses.KEY_ENTER, 10, 13]: return selection == 0


def show_menu(win: curses.window, selection: int, menu: List[str]):
    """Displays the main menu with a title.

    Args:
        win (curses.window): The curses window to draw in.
        selection (int): The index of the currently selected menu item.
        menu (list): The list of menu item strings.
    """
    h, w = win.getmaxyx()
    win.clear()
    title_art = pyfiglet.figlet_format("NITW Patcher", font=FIGLET_FONT, width=w)
    title_lines = title_art.strip().splitlines()
    title_height = len(title_lines)
    title_width = len(title_lines[0]) if title_lines else 0

    if w < title_width + 4 or h < title_height + len(menu) + 4:
        title = "NITW Patcher"
        start_y, menu_y = 2, 4
        win.addstr(start_y, w // 2 - len(title) // 2, title, curses.A_BOLD)
    else:
        start_y = h // 2 - (title_height + len(menu)) // 2
        if start_y < 1: start_y = 1
        for i, line in enumerate(title_lines):
            win.addstr(start_y + i, w // 2 - len(line) // 2, line, curses.A_BOLD)
        menu_y = start_y + title_height + 1

    for i, row in enumerate(menu):
        x, y = w // 2 - len(row) // 2, menu_y + i
        if i == selection: win.attron(curses.color_pair(COLOR_PAIR_SELECTED))
        win.addstr(y, x, row)
        if i == selection: win.attroff(curses.color_pair(COLOR_PAIR_SELECTED))
    win.refresh()


def show_message(win: curses.window, message: str):
    """Displays a centered message and waits for a key press.

    Args:
        win (curses.window): The curses window to draw in.
        message (str): The message to display.
    """
    h, w = win.getmaxyx(); win.clear()
    win.addstr(h // 2, w // 2 - len(message) // 2, message)
    win.refresh(); win.getch()


def main_loop(stdscr: curses.window):
    """Sets up and runs the main TUI event loop.

    Args:
        stdscr (curses.window): The standard screen object provided by curses.wrapper.
    """
    curses.curs_set(0)
    curses.init_pair(COLOR_PAIR_SELECTED, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(COLOR_PAIR_CANCEL, curses.COLOR_WHITE, curses.COLOR_RED)
    
    h, w = stdscr.getmaxyx()
    win = curses.newwin(h, w, 0, 0)
    content_win = win.subwin(h - 2, w - 2, 1, 1)
    content_win.keypad(True)

    menu = ["Extract", "Normalize", "Validate", "Patch", "Exit"]
    actions = {0: extract, 1: normalize, 2: validate, 3: patch}
    selection = 0

    while True:
        win.box(); win.refresh()
        show_menu(content_win, selection, menu)
        
        key = content_win.getch()
        if key == curses.KEY_UP: selection = (selection - 1) % len(menu)
        elif key == curses.KEY_DOWN: selection = (selection + 1) % len(menu)
        elif key in [curses.KEY_ENTER, 10, 13]:
            if selection == len(menu) - 1: break
            
            status = ""
            try:
                action_func = actions[selection]
                progress_callback = lambda current, total, msg: show_progress(content_win, current, total, msg)
                action_generator = action_func(on_progress=progress_callback) if action_func in [extract, validate, patch] else action_func()

                for result in action_generator:
                    match result:
                        case ("write_files", addresses, names, phrases):
                            files_to_write = [(ADDRESSES_FILE, addresses), (NAMES_FILE, names), (ORIGINAL_PHRASES_FILE, phrases)]
                            for filepath, data in files_to_write:
                                with open(filepath, "w", encoding="utf-8") as f: f.write("\n".join(data))
                            if os.path.exists(TRANSLATED_PHRASES_FILE):
                                if confirm(content_win, f"'{TRANSLATED_PHRASES_FILE}' exists. Overwrite?"):
                                    shutil.copy(ORIGINAL_PHRASES_FILE, TRANSLATED_PHRASES_FILE)
                            else: shutil.copy(ORIGINAL_PHRASES_FILE, TRANSLATED_PHRASES_FILE)
                        
                        case ("confirm_normalize", changes, lines):
                            if confirm(content_win, f"Found {changes} lines to normalize. Apply changes?"):
                                with open(TRANSLATED_PHRASES_FILE, 'w', encoding='utf-8') as f: f.write("\n".join(lines))
                                status = f"Normalized {changes} lines."
                            else: status = "Normalization cancelled."
                        
                        case ("confirm_view_errors", errors):
                            if confirm(content_win, f"Found {len(errors)} validation errors. View them?"):
                                view_errors(content_win, "Keyword Validation Errors", errors)
                                status = "Finished reviewing errors."
                            else: status = "Validation finished. Errors not viewed."
                        
                        case str(message):
                            status = message
            
            except Exception as e:
                status = f"An unexpected error occurred: {e}"
            
            show_message(content_win, status)


if __name__ == "__main__":
    try:
        curses.wrapper(main_loop)
    except curses.error as e:
        print(f"Terminal error: {e}\nOn Windows, ensure 'windows-curses' is installed.")
    except ImportError:
        print("Error: 'pyfiglet' is required.\nPlease run: pip install pyfiglet")
    except KeyboardInterrupt:
        print("\nProgram exited.")
