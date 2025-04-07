import argparse
import os

import nltk
import numpy as np

from utils.helper import (
    calculate_bleu,
    calculate_cer,
    calculate_edit_distance,
    calculate_mer,
    calculate_wer,
)

# Download necessary nltk packages
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def process_folders(ground_truth_folder, ocr_result_folder):
    """Process all files in the folder structure and calculate metrics."""
    
    # List of subfolders to process
    subfolders = ['answer', 'question', 'solution']
    
    all_metrics = {
        'wer': [],
        'cer': [],
        'mer': [],
        'bleu': [],
        'edit_distance': []
    }
    
    # Track files processed and errors
    files_processed = 0
    errors = 0
    
    # Process each subfolder
    for subfolder in subfolders:
        gt_subfolder = os.path.join(ground_truth_folder, subfolder)
        ocr_subfolder = os.path.join(ocr_result_folder, subfolder)
        
        # Skip if either subfolder doesn't exist
        if not os.path.exists(gt_subfolder) or not os.path.exists(ocr_subfolder):
            print(f"Skipping subfolder {subfolder} - not found in both locations")
            continue
        
        # Get all files in ground truth subfolder
        gt_files = [f for f in os.listdir(gt_subfolder) if f.endswith('.txt')]
        
        # Process each file
        subfolder_metrics = {
            'wer': [],
            'cer': [],
            'mer': [],
            'bleu': [],
            'edit_distance': []
        }
        
        for file_name in gt_files:
            gt_file_path = os.path.join(gt_subfolder, file_name)
            ocr_file_path = os.path.join(ocr_subfolder, file_name)
            
            # Skip if OCR file doesn't exist
            if not os.path.exists(ocr_file_path):
                print(f"Skipping file {file_name} - not found in OCR results")
                errors += 1
                continue
            
            try:
                # Read files
                with open(gt_file_path, 'r', encoding='utf-8') as f:
                    gt_text = f.read()
                with open(ocr_file_path, 'r', encoding='utf-8') as f:
                    ocr_text = f.read()
                
                # Calculate metrics
                wer = calculate_wer(gt_text, ocr_text)
                cer = calculate_cer(gt_text, ocr_text)
                mer = calculate_mer(gt_text, ocr_text)
                bleu = calculate_bleu(gt_text, ocr_text)
                edit_dist = calculate_edit_distance(gt_text, ocr_text)
                
                # Add to subfolder metrics
                subfolder_metrics['wer'].append(wer)
                subfolder_metrics['cer'].append(cer)
                subfolder_metrics['mer'].append(mer)
                subfolder_metrics['bleu'].append(bleu)
                subfolder_metrics['edit_distance'].append(edit_dist)
                
                # Add to overall metrics
                all_metrics['wer'].append(wer)
                all_metrics['cer'].append(cer)
                all_metrics['mer'].append(mer)
                all_metrics['bleu'].append(bleu)
                all_metrics['edit_distance'].append(edit_dist)
                
                files_processed += 1
            
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                errors += 1
        
        # Print subfolder metrics
        print(f"\n--- Metrics for {subfolder} ---")
        print(f"Files processed: {len(subfolder_metrics['wer'])}")
        
        if len(subfolder_metrics['wer']) > 0:
            print(f"WER: {np.mean(subfolder_metrics['wer']):.4f}")
            print(f"CER: {np.mean(subfolder_metrics['cer']):.4f}")
            print(f"MER: {np.mean(subfolder_metrics['mer']):.4f}")
            print(f"BLEU: {np.mean(subfolder_metrics['bleu']):.4f}")
            print(f"Avg Edit Distance: {np.mean(subfolder_metrics['edit_distance']):.1f}")
    
    # Print overall metrics
    print("\n--- Overall Metrics ---")
    print(f"Total files processed: {files_processed}")
    print(f"Errors encountered: {errors}")
    
    if files_processed > 0:
        print(f"Overall WER: {np.mean(all_metrics['wer']):.4f}")
        print(f"Overall CER: {np.mean(all_metrics['cer']):.4f}")
        print(f"Overall MER: {np.mean(all_metrics['mer']):.4f}")
        print(f"Overall BLEU: {np.mean(all_metrics['bleu']):.4f}")
        print(f"Overall Avg Edit Distance: {np.mean(all_metrics['edit_distance']):.1f}")
    
    return all_metrics, files_processed, errors

def save_results_to_csv(metrics, output_file):
    """Save detailed results to CSV file."""
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['Metric', 'Mean', 'Median', 'Min', 'Max', 'Std Dev'])
        
        # Write each metric's stats
        for metric_name, values in metrics.items():
            if values:
                writer.writerow([
                    metric_name,
                    f"{np.mean(values):.6f}",
                    f"{np.median(values):.6f}",
                    f"{np.min(values):.6f}",
                    f"{np.max(values):.6f}",
                    f"{np.std(values):.6f}"
                ])
            else:
                writer.writerow([metric_name, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])

def main():
    parser = argparse.ArgumentParser(description='Evaluate OCR results against ground truth')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth folder')
    parser.add_argument('--ocr-result', required=True, help='Path to OCR result folder')
    parser.add_argument('--output', default='ocr_evaluation_results.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    print(f"Comparing OCR results in {args.ocr_result} against ground truth in {args.ground_truth}")
    
    # Process the folders
    metrics, files_processed, errors = process_folders(args.ground_truth, args.ocr_result)
    
    # Save results to CSV if files were processed
    if files_processed > 0:
        save_results_to_csv(metrics, args.output)
        print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main()
