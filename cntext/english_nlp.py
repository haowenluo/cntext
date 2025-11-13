"""
Enhanced English NLP utilities for better text processing.

This module provides improved English text tokenization, lemmatization,
and preprocessing using spaCy (if available) with fallback to NLTK.

Features:
- Better tokenization than simple text.split()
- Optional lemmatization (running → run, better → good)
- Punctuation handling
- Automatic spaCy/NLTK detection
"""

import re
import string
from typing import List, Optional

# Try to import spaCy
try:
    import spacy
    try:
        # Try to load the small English model
        _nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        # Model not downloaded
        SPACY_AVAILABLE = False
        _nlp = None
except ImportError:
    # spaCy not installed
    SPACY_AVAILABLE = False
    _nlp = None

# Fallback to NLTK
from nltk.tokenize import word_tokenize
try:
    from nltk.stem import WordNetLemmatizer
    _lemmatizer = WordNetLemmatizer()
    NLTK_LEMMATIZER_AVAILABLE = True
except:
    NLTK_LEMMATIZER_AVAILABLE = False
    _lemmatizer = None


def is_spacy_available() -> bool:
    """
    Check if spaCy is available for use.

    Returns:
        bool: True if spaCy and en_core_web_sm model are available
    """
    return SPACY_AVAILABLE


def tokenize_english(text: str,
                     lemmatize: bool = False,
                     remove_punct: bool = True,
                     lowercase: bool = True) -> List[str]:
    """
    Tokenize English text with optional lemmatization and preprocessing.

    Uses spaCy if available (better quality), otherwise falls back to NLTK.

    Args:
        text (str): Input text to tokenize
        lemmatize (bool): Whether to lemmatize tokens (default: False)
        remove_punct (bool): Whether to remove punctuation (default: True)
        lowercase (bool): Whether to convert to lowercase (default: True)

    Returns:
        List[str]: List of tokens

    Examples:
        >>> tokenize_english("The companies are running quickly!")
        ['the', 'companies', 'are', 'running', 'quickly']

        >>> tokenize_english("The companies are running quickly!", lemmatize=True)
        ['the', 'company', 'be', 'run', 'quickly']
    """
    if not text or not text.strip():
        return []

    # Use spaCy if available
    if SPACY_AVAILABLE and _nlp:
        doc = _nlp(text)

        if lemmatize:
            tokens = [token.lemma_ for token in doc]
        else:
            tokens = [token.text for token in doc]

        # Filter punctuation if requested
        if remove_punct:
            if lemmatize:
                tokens = [token.lemma_ for token in doc
                         if not token.is_punct and not token.is_space]
            else:
                tokens = [token.text for token in doc
                         if not token.is_punct and not token.is_space]

        # Lowercase if requested
        if lowercase:
            tokens = [t.lower() for t in tokens]

        return tokens

    # Fallback to NLTK
    else:
        # Basic tokenization
        tokens = word_tokenize(text)

        # Remove punctuation if requested
        if remove_punct:
            tokens = [t for t in tokens if t not in string.punctuation]

        # Lemmatize using NLTK if requested and available
        if lemmatize and NLTK_LEMMATIZER_AVAILABLE and _lemmatizer:
            tokens = [_lemmatizer.lemmatize(t) for t in tokens]

        # Lowercase if requested
        if lowercase:
            tokens = [t.lower() for t in tokens]

        return tokens


def preprocess_english(text: str,
                      lemmatize: bool = False,
                      remove_punct: bool = True,
                      remove_numbers: bool = False,
                      number_replacement: str = '_num_',
                      min_length: int = 1,
                      lowercase: bool = True,
                      stopwords: Optional[set] = None) -> List[str]:
    """
    Comprehensive preprocessing for English text.

    This function provides a complete preprocessing pipeline:
    1. Normalize whitespace
    2. Handle numbers (remove or replace)
    3. Tokenize (with optional lemmatization)
    4. Remove stopwords
    5. Filter by length

    Args:
        text (str): Input text to preprocess
        lemmatize (bool): Whether to lemmatize tokens (default: False)
        remove_punct (bool): Whether to remove punctuation (default: True)
        remove_numbers (bool): Whether to completely remove numbers (default: False)
        number_replacement (str): What to replace numbers with if not removing (default: '_num_')
        min_length (int): Minimum token length to keep (default: 1)
        lowercase (bool): Whether to convert to lowercase (default: True)
        stopwords (set, optional): Set of stopwords to remove (default: None)

    Returns:
        List[str]: List of preprocessed tokens

    Examples:
        >>> preprocess_english("The company's revenue grew 25% in 2024!")
        ['company', 's', 'revenue', 'grew', '_num_', 'in', '_num_']

        >>> preprocess_english("The company's revenue grew 25% in 2024!",
        ...                   lemmatize=True, remove_numbers=True)
        ['the', 'company', 's', 'revenue', 'grow', 'in']
    """
    if not text or not text.strip():
        return []

    # 1. Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # 2. Handle numbers
    if remove_numbers:
        text = re.sub(r'\b\d+\.?\d*\b', '', text)
    elif number_replacement:
        text = re.sub(r'\b\d+\.?\d*\b', f' {number_replacement} ', text)

    # 3. Tokenize with optional lemmatization
    tokens = tokenize_english(text,
                             lemmatize=lemmatize,
                             remove_punct=remove_punct,
                             lowercase=lowercase)

    # 4. Remove stopwords if provided
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]

    # 5. Filter by minimum length
    if min_length > 0:
        tokens = [t for t in tokens if len(t) >= min_length]

    return tokens


def get_backend_info() -> dict:
    """
    Get information about which NLP backend is being used.

    Returns:
        dict: Information about available backends and current configuration

    Example:
        >>> info = get_backend_info()
        >>> print(info['backend'])
        'spacy'  # or 'nltk'
    """
    backend = 'spacy' if SPACY_AVAILABLE else 'nltk'

    return {
        'backend': backend,
        'spacy_available': SPACY_AVAILABLE,
        'nltk_lemmatizer_available': NLTK_LEMMATIZER_AVAILABLE,
        'spacy_model': 'en_core_web_sm' if SPACY_AVAILABLE else None,
        'recommendation': 'Install spacy and download en_core_web_sm for better English NLP'
                         if not SPACY_AVAILABLE else 'Using spaCy for optimal performance'
    }


# Convenience function for quick tokenization
def tokenize(text: str, lemmatize: bool = False) -> List[str]:
    """
    Quick tokenization function with sensible defaults.

    Args:
        text (str): Input text
        lemmatize (bool): Whether to lemmatize (default: False)

    Returns:
        List[str]: List of tokens
    """
    return tokenize_english(text, lemmatize=lemmatize, remove_punct=True, lowercase=True)


if __name__ == '__main__':
    # Test the module
    print("English NLP Backend Info:")
    print(get_backend_info())
    print()

    # Test text
    test_text = "The companies are running innovative programs. Revenue increased by 25% in 2024!"

    print(f"Original text: {test_text}")
    print()

    print("Basic tokenization:")
    print(tokenize_english(test_text, lemmatize=False))
    print()

    print("With lemmatization:")
    print(tokenize_english(test_text, lemmatize=True))
    print()

    print("Full preprocessing (with lemmatization):")
    print(preprocess_english(test_text, lemmatize=True, remove_numbers=False))
