# NITW Patcher

A minimalist, terminal-based utility for extracting, managing, and patching dialogue files for the game *Night in the Woods*.

This tool provides a simple interface for translators to work with game files, offering a complete workflow from extraction to final patching.

## Features

*   **Extraction:** Pulls phrases into clean text file.
*   **Normalization:** Fixes common capitalization and punctuation errors.
*   **Validation:** Ensures key terms are translated consistently via custom rules.
*   **Patching:** Injects translated text back into the game files.

## Requirements

*   Python 3.10+
*   Original game dialogue files

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-folder>
    ```

2.  **Install dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare game files:**
    Place your original, unedited game dialogue files into the `original/` directory.

## Usage

Run the main script from your terminal:

```bash
python main.py
```

Use the **UP/DOWN** arrow keys to navigate and **ENTER** to select an option.

## Workflow

The program guides you through a clear, step-by-step process. The file structure it creates is central to the workflow:

```
.
├── original/           # 1. Place your original game files here first.
│   └── game_file_1.txt
│
├── extract/            # 2. Created by the "Extract" option.
│   ├── original.txt    #    - The source text for reference.
│   ├── translated.txt  #    - EDIT THIS FILE with your translations.
│   ├── names.txt       #    - A helpful list of corresponding character names.
│   └── addresses.txt   #    - Internal data for the patcher. Do not edit.
│
├── patched/            # 4. Created by the "Patch" option.
│   └── game_file_1.txt #    - Your final, modded files, ready to be used.
│
├── validation_rules.json # 3. (Optional) Add your keyword rules here before running "Validate".
│
└── main.py             # Run this script to start the TUI.
