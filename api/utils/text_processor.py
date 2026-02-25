"""
Text processing utilities for cleaning OCR output and extracted text.

This module provides functions for:
- Cleaning OCR output (removing noise, normalizing formatting)
- Removing non-content elements (hashtags, repeated chars, emojis)
- Confidence filtering
- Text merging and deduplication
"""

import re
import string
import unicodedata
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CleanedText:
    """Result of text cleaning operation"""
    original: str
    cleaned: str
    confidence: float
    removed_elements: dict
    is_valid: bool


class TextCleaner:
    """Comprehensive text cleaning for OCR output"""
    
    # Thresholds
    MIN_TEXT_LENGTH = 3  # Minimum characters for valid text
    MAX_REPEATED_CHAR_RATIO = 0.3  # Max ratio of repeated chars
    MIN_CONFIDENCE_THRESHOLD = 0.5  # Minimum OCR confidence
    
    # Patterns for noise removal
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F700-\U0001F77F"  # Alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric shapes
        "\U0001F800-\U0001F8FF"  # Supplemental arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental symbols & pictographs
        "\U0001FA00-\U0001FA6F"  # Chess symbols
        "\U0001FA70-\U0001FAFF"  # Symbols & pictographs extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+"
    )
    
    # Hashtags and @ mentions
    HASHTAG_PATTERN = re.compile(r'#\w+')
    MENTION_PATTERN = re.compile(r'@\w+')
    
    # URLs
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    # Multiple spaces, newlines
    WHITESPACE_PATTERN = re.compile(r'\s+')
    
    # Common UI/social media text to filter
    COMMON_UI_PATTERNS = [
        r'\blike\b',
        r'\bcomment\b',
        r'\bshare\b',
        r'\bfollow\b',
        r'\bsubscribe\b',
        r'\bfaithful\b',
        r'\btrending\b',
        r'\bview\b',
        r'\bviews\b',
    ]

    @staticmethod
    def remove_emojis(text: str) -> str:
        """Remove emoji characters from text"""
        return TextCleaner.EMOJI_PATTERN.sub('', text)

    @staticmethod
    def remove_hashtags(text: str, keep: bool = False) -> str:
        """Remove or keep hashtags"""
        if keep:
            return TextCleaner.HASHTAG_PATTERN.sub(lambda m: m.group(0)[1:], text)
        return TextCleaner.HASHTAG_PATTERN.sub('', text)

    @staticmethod
    def remove_mentions(text: str) -> str:
        """Remove @ mentions"""
        return TextCleaner.MENTION_PATTERN.sub('', text)

    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs"""
        return TextCleaner.URL_PATTERN.sub('', text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace: collapse multiple spaces, normalize newlines"""
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = TextCleaner.WHITESPACE_PATTERN.sub(' ', text)
        return text.strip()

    @staticmethod
    def remove_repeated_characters(text: str, max_repetition: int = 3) -> str:
        """Remove excessive character repetition (e.g., 'hellooooo' -> 'hello')"""
        return re.sub(r'(.)\1{' + str(max_repetition) + ',}', r'\1' * max_repetition, text)

    @staticmethod
    def check_repeated_char_ratio(text: str, max_ratio: float = 0.3) -> bool:
        """Check if text has too many repeated characters (likely noise)"""
        if len(text) < 1:
            return False
        
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        max_count = max(char_counts.values())
        ratio = max_count / len(text)
        
        return ratio < max_ratio  # Return True if acceptable (ratio below max)

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode to NFC form"""
        return unicodedata.normalize('NFC', text)

    @staticmethod
    def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
        """Remove special/control characters"""
        if keep_punctuation:
            # Keep alphanumeric, whitespace, and common punctuation
            allowed = string.ascii_letters + string.digits + ' .,!?;:-\'"'
            return ''.join(char for char in text if char in allowed)
        else:
            return re.sub(r'[^\w\s]', '', text)

    @staticmethod
    def is_valid_text(text: str, min_length: int = 3, min_alphanumeric: float = 0.3) -> bool:
        """
        Check if text is actually valid content.
        
        Args:
            text: Text to validate
            min_length: Minimum character length
            min_alphanumeric: Minimum ratio of alphanumeric characters
            
        Returns:
            True if text appears to be valid content
        """
        if len(text) < min_length:
            return False
        
        alphanumeric_count = sum(1 for c in text if c.isalnum())
        alphanumeric_ratio = alphanumeric_count / len(text)
        
        if alphanumeric_ratio < min_alphanumeric:
            return False
        
        return True

    @classmethod
    def clean(
        cls,
        text: str,
        confidence: float = 1.0,
        remove_hashtags: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_emojis: bool = True,
        min_confidence: float = None,
    ) -> CleanedText:
        """
        Comprehensive text cleaning for OCR output.
        
        Args:
            text: Raw text to clean
            confidence: OCR confidence score (0-1)
            remove_hashtags: Whether to remove hashtags
            remove_urls: Whether to remove URLs
            remove_mentions: Whether to remove mentions
            remove_emojis: Whether to remove emojis
            min_confidence: Minimum confidence threshold
            
        Returns:
            CleanedText object with cleaned text and metadata
        """
        removed = {
            'emojis': 0,
            'hashtags': 0,
            'urls': 0,
            'mentions': 0,
            'repeated_chars': 0,
            'whitespace_normalized': False,
        }
        
        original = text
        cleaned = text
        
        # Check minimum confidence
        if min_confidence is None:
            min_confidence = cls.MIN_CONFIDENCE_THRESHOLD
        
        if confidence < min_confidence:
            logger.warning(f"Text confidence {confidence} below threshold {min_confidence}")
            return CleanedText(
                original=original,
                cleaned="",
                confidence=confidence,
                removed_elements=removed,
                is_valid=False,
            )
        
        # Step 1: Normalize unicode
        cleaned = cls.normalize_unicode(cleaned)
        
        # Step 2: Remove emojis
        if remove_emojis:
            before = len(cleaned)
            cleaned = cls.remove_emojis(cleaned)
            removed['emojis'] = before - len(cleaned)
        
        # Step 3: Remove URLs
        if remove_urls:
            before = len(cleaned)
            cleaned = cls.remove_urls(cleaned)
            removed['urls'] = before - len(cleaned)
        
        # Step 4: Remove mentions
        if remove_mentions:
            before = len(cleaned)
            cleaned = cls.remove_mentions(cleaned)
            removed['mentions'] = before - len(cleaned)
        
        # Step 5: Remove hashtags
        if remove_hashtags:
            before = len(cleaned)
            cleaned = cls.remove_hashtags(cleaned)
            removed['hashtags'] = before - len(cleaned)
        
        # Step 6: Remove excessive character repetition
        before = len(cleaned)
        cleaned = cls.remove_repeated_characters(cleaned)
        removed['repeated_chars'] = before - len(cleaned)
        
        # Step 7: Normalize whitespace
        before_ws = cleaned
        cleaned = cls.normalize_whitespace(cleaned)
        removed['whitespace_normalized'] = before_ws != cleaned
        
        # Step 8: Final validation
        is_valid = (
            cls.is_valid_text(cleaned) and
            cls.check_repeated_char_ratio(cleaned, cls.MAX_REPEATED_CHAR_RATIO) and
            len(cleaned.strip()) >= cls.MIN_TEXT_LENGTH
        )
        
        logger.info(f"Text cleaning: {len(original)} -> {len(cleaned)} chars, valid={is_valid}")
        
        return CleanedText(
            original=original,
            cleaned=cleaned,
            confidence=confidence,
            removed_elements=removed,
            is_valid=is_valid,
        )


def merge_text_blocks(text_blocks: List[str], remove_duplicates: bool = True) -> str:
    """
    Merge multiple text blocks into a single text.
    
    Args:
        text_blocks: List of text strings
        remove_duplicates: Whether to remove duplicate lines
        
    Returns:
        Merged and cleaned text
    """
    if not text_blocks:
        return ""
    
    # Filter None/empty values
    valid_blocks = [str(block).strip() for block in text_blocks if block]
    
    if remove_duplicates:
        seen = set()
        unique_blocks = []
        for block in valid_blocks:
            if block not in seen:
                unique_blocks.append(block)
                seen.add(block)
        valid_blocks = unique_blocks
    
    merged = '\n'.join(valid_blocks)
    return TextCleaner.normalize_whitespace(merged)


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from cleaned text.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Split on common sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def get_text_statistics(text: str) -> dict:
    """
    Get basic statistics about the text.
    
    Returns:
        Dictionary with text stats
    """
    words = text.split()
    sentences = extract_sentences(text)
    
    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': len(text) / len(words) if words else 0,
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
    }
