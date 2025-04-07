# OCR Evaluation Tool

A Python utility for evaluating Optical Character Recognition (OCR) results against ground truth files using multiple metrics.

## Overview

This tool compares OCR-generated text files against corresponding ground truth files to calculate accuracy metrics including:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Match Error Rate (MER)
- BLEU Score
- Edit Distance (Levenshtein)

The script processes a specific folder structure where both ground truth and OCR results share the same organization pattern, with matching filenames in corresponding directories.

## Folder Structure

The script expects the following folder structure:
```
ground_truth/
├── answer/
│   ├── file1.txt
│   ├── file2.txt
│   └── ...
├── question/
│   ├── file1.txt
│   ├── file2.txt
│   └── ...
└── solution/
    ├── file1.txt
    ├── file2.txt
    └── ...

ocr_result/
├── answer/
│   ├── file1.txt
│   ├── file2.txt
│   └── ...
├── question/
│   ├── file1.txt
│   ├── file2.txt
│   └── ...
└── solution/
    ├── file1.txt
    ├── file2.txt
    └── ...
```

## Requirements

- Python 3.6+
- Dependencies:
  - nltk
  - numpy
  - python-Levenshtein

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/SaimonDahal-02/ocr-evaluation.git
   cd ocr-evaluation
   ```

2. Install required packages using uv as dependency management:
   ```bash
   pip install uv
   uv venv
   source .venv/bin/activate
   uv sync
   ```

## Usage

Run the script using the following command:

```bash
python ocr_evaluation.py --ground-truth /path/to/ground_truth --ocr-result /path/to/ocr_result --output results.csv
```

### Arguments

- `--ground-truth`: Path to the folder containing ground truth text files (required)
- `--ocr-result`: Path to the folder containing OCR result text files (required)
- `--output`: Path to save the output CSV file (default: `ocr_evaluation_results.csv`)

## Output

The script provides:

1. Console output with summary statistics for each subfolder and overall results
2. A CSV file with detailed metrics including:
   - Mean, median, min, max values
   - Standard deviation for each metric

Example console output:
```
--- Metrics for answer ---
Files processed: 25
WER: 0.0845
CER: 0.0312
MER: 0.0723
BLEU: 0.8956
Avg Edit Distance: 12.4

--- Metrics for question ---
Files processed: 30
WER: 0.0632
CER: 0.0257
MER: 0.0541
BLEU: 0.9125
Avg Edit Distance: 8.7

--- Overall Metrics ---
Total files processed: 55
Errors encountered: 2
Overall WER: 0.0728
Overall CER: 0.0281
Overall MER: 0.0621
Overall BLEU: 0.9052
Overall Avg Edit Distance: 10.3
```

## Metrics Explained

- **Character Error Rate (CER)**: The percentage of characters that were incorrectly recognized, calculated as the edit distance divided by the total number of characters.
- **Word Error Rate (WER)**: The percentage of words that were incorrectly recognized, calculated as the edit distance (at word level) divided by the total number of words.
- **Match Error Rate (MER)**: A measure that considers word ordering errors, calculated as 1 minus the ratio of matching words to the maximum length.
- **BLEU Score**: A metric from machine translation that measures the overlap of n-grams between ground truth and OCR text. Higher values indicate better performance.
- **Edit Distance**: Raw Levenshtein distance between strings, representing the minimum number of single-character edits required to change one string into the other.

