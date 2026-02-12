"""
CEO narcissism metrics based on first-person pronoun usage.

Implements the First-Person Singular Pronoun (FPSP) ratio used in
financial research to measure CEO narcissism from earnings call
transcripts.

Metrics:
    - FPSP Ratio: singular / (singular + plural)
      Standard measure from Aktas et al. (2016).
    - FPSP per 1000 words: (singular_count / total_words) * 1000
      Alternative normalisation used in some studies.

References:
    Aktas, N., de Bodt, E., Bollaert, H., & Roll, R. (2016).
        CEO narcissism and the takeover process. Journal of Financial
        and Quantitative Analysis, 51(1), 113-137.
    Capalbo, F., Frino, A., Lim, M. Y., Mollica, V., & Palumbo, R. (2018).
        CEO narcissism and earnings management. Working Paper.
"""

import re
import pandas as pd
from typing import Dict, Optional

from ..io.dict import read_yaml_dict


# Default pronoun sets (matching the YAML dictionary)
SINGULAR_PRONOUNS = {'i', 'me', 'my', 'mine', 'myself'}
PLURAL_PRONOUNS = {'we', 'us', 'our', 'ours', 'ourselves'}


def _tokenize_for_pronouns(text: str):
    """
    Tokenize text for pronoun counting.

    Uses a regex tokenizer that preserves single-letter words like "I"
    while splitting on non-alphabetic boundaries. Does NOT remove
    stopwords (pronouns ARE the target).

    Returns:
        list[str]: Lowercased tokens.
    """
    # Split on non-alphabetic characters, keeping apostrophe-based
    # contractions together (e.g. "I'm" -> ["i", "m"])
    rgx = re.compile(r"(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)")
    tokens = re.split(rgx, text.lower())
    return [t for t in tokens if t]


def fpsp_ratio(text: str,
               singular: Optional[set] = None,
               plural: Optional[set] = None,
               return_series: bool = False):
    """
    Calculate the First-Person Singular Pronoun (FPSP) narcissism ratio.

    Formula (Aktas et al., 2016):
        FPSP_Ratio = Σ(singular) / (Σ(singular) + Σ(plural))

    Also returns:
        - singular_count, plural_count: raw counts
        - fpsp_per_1000: singular count per 1,000 words
        - fppp_per_1000: plural count per 1,000 words
        - word_count: total number of tokens in text

    Args:
        text (str): Text to analyse (typically CEO Q&A speech).
        singular (set, optional): Custom set of singular pronouns.
            Defaults to {'i', 'me', 'my', 'mine', 'myself'}.
        plural (set, optional): Custom set of plural pronouns.
            Defaults to {'we', 'us', 'our', 'ours', 'ourselves'}.
        return_series (bool): If True, return pd.Series instead of dict.

    Returns:
        dict or pd.Series with keys:
            singular_count (int)
            plural_count (int)
            fpsp_ratio (float): Σ(S) / (Σ(S) + Σ(P)), or NaN if
                denominator is zero.
            fpsp_per_1000 (float): singular_count / word_count * 1000
            fppp_per_1000 (float): plural_count / word_count * 1000
            word_count (int)

    Example:
        >>> text = "I believe we can do this. My vision is clear."
        >>> fpsp_ratio(text)
        {'singular_count': 2, 'plural_count': 1, 'fpsp_ratio': 0.6667,
         'fpsp_per_1000': 181.8182, 'fppp_per_1000': 90.9091, 'word_count': 11}
    """
    if singular is None:
        singular = SINGULAR_PRONOUNS
    if plural is None:
        plural = PLURAL_PRONOUNS

    tokens = _tokenize_for_pronouns(text)
    word_count = len(tokens)

    singular_count = sum(1 for t in tokens if t in singular)
    plural_count = sum(1 for t in tokens if t in plural)

    denom = singular_count + plural_count
    if denom > 0:
        ratio = round(singular_count / denom, 4)
    else:
        ratio = float('nan')

    if word_count > 0:
        fpsp_per_1000 = round(singular_count / word_count * 1000, 4)
        fppp_per_1000 = round(plural_count / word_count * 1000, 4)
    else:
        fpsp_per_1000 = float('nan')
        fppp_per_1000 = float('nan')

    result = {
        'singular_count': singular_count,
        'plural_count': plural_count,
        'fpsp_ratio': ratio,
        'fpsp_per_1000': fpsp_per_1000,
        'fppp_per_1000': fppp_per_1000,
        'word_count': word_count,
    }

    if return_series:
        return pd.Series(result)
    return result


def fpsp_ratio_from_dict(text: str,
                         dict_file: str = 'en_common_Pronouns.yaml',
                         is_builtin: bool = True,
                         return_series: bool = False):
    """
    Calculate FPSP ratio using pronoun lists loaded from a YAML dictionary.

    This is a convenience wrapper that loads the pronoun dictionary via
    cntext's standard read_yaml_dict() function.

    Args:
        text (str): Text to analyse.
        dict_file (str): Path to YAML dictionary file.
            Defaults to the built-in 'en_common_Pronouns.yaml'.
        is_builtin (bool): Whether dict_file is a built-in cntext dictionary.
        return_series (bool): If True, return pd.Series.

    Returns:
        dict or pd.Series (same as fpsp_ratio).
    """
    d = read_yaml_dict(dict_file, is_builtin=is_builtin)
    singular = set(d['Dictionary']['singular'])
    plural = set(d['Dictionary']['plural'])
    return fpsp_ratio(text, singular=singular, plural=plural,
                      return_series=return_series)
