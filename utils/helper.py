import re
import string
from collections import Counter

import Levenshtein
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def tokenize_text(text):
    """Tokenize text into words."""
    # Remove punctuation and lowercase
    text = text.lower()
    text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), ' ', text)
    # Split on whitespace and filter empty strings
    return [word for word in text.split() if word]

def char_tokenize(text):
    """Tokenize text into characters."""
    return list(text.replace(" ", ""))

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate.
    WER = (S + D + I) / N
    where:
    S = substitutions, D = deletions, I = insertions, N = number of words in reference
    """
    # Tokenize into words
    ref_words = tokenize_text(reference)
    hyp_words = tokenize_text(hypothesis)
    
    # Calculate edit distance
    edit_distance = Levenshtein.distance(' '.join(ref_words), ' '.join(hyp_words))
    
    # Count words in reference
    if len(ref_words) == 0:
        return 0 if len(hyp_words) == 0 else 1
    
    return edit_distance / len(ref_words)

def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate.
    CER = (S + D + I) / N
    where:
    S = substitutions, D = deletions, I = insertions, N = number of characters in reference
    """
    # Remove spaces for character comparison
    ref_chars = char_tokenize(reference)
    hyp_chars = char_tokenize(hypothesis)
    
    # Calculate edit distance
    edit_distance = Levenshtein.distance(''.join(ref_chars), ''.join(hyp_chars))
    
    # Count characters in reference
    if len(ref_chars) == 0:
        return 0 if len(hyp_chars) == 0 else 1
    
    return edit_distance / len(ref_chars)

def calculate_mer(reference, hypothesis):
    """
    Calculate Match Error Rate.
    MER = 1 - (number of matches / max(len(reference), len(hypothesis)))
    """
    ref_words = tokenize_text(reference)
    hyp_words = tokenize_text(hypothesis)
    
    # Count matching words
    ref_counter = Counter(ref_words)
    hyp_counter = Counter(hyp_words)
    
    # Count the matches (intersection of the two multisets)
    matches = sum((ref_counter & hyp_counter).values())
    
    # Calculate MER
    max_len = max(len(ref_words), len(hyp_words))
    if max_len == 0:
        return 0
    
    return 1 - (matches / max_len)

def calculate_bleu(reference, hypothesis):
    """Calculate BLEU score."""
    ref_tokens = [tokenize_text(reference)]
    hyp_tokens = tokenize_text(hypothesis)
    
    # If either is empty, return 0
    if not ref_tokens[0] or not hyp_tokens:
        return 0
    
    # Use smoothing to avoid 0 scores when there are no n-gram matches
    smoothie = SmoothingFunction().method1
    
    # Calculate BLEU-4 score
    try:
        return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        return 0

def calculate_edit_distance(reference, hypothesis):
    """Calculate raw Levenshtein edit distance between strings."""
    return Levenshtein.distance(reference, hypothesis)
