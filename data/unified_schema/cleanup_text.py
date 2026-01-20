from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Dict, Optional, Literal, Union
from urllib.parse import urlparse

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

# regex patterns
_MENTION_RE = re.compile(r"@[\w.\-]+")
_HASHTAG_RE = re.compile(r"#([\w\-]+)")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEAT_PUNCT_RE = re.compile(r"([!? ,])\1{1,}")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

def strip_boilerplate_heuristic(text: str) -> str:
    kept = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue

        url_count = len(_URL_RE.findall(s))
        letters = sum(ch.isalpha() for ch in s)
        nonspace = sum(not ch.isspace() for ch in s)
        punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in s)
        sep_count = sum(s.count(c) for c in ["|", ">", "•", "»"])

        # generic filters
        if len(s) < 30 and url_count == 0 and letters < 15:
            continue
        if url_count >= 2:
            continue
        if nonspace and (punct / nonspace) > 0.35 and letters < 20:
            continue
        if sep_count >= 5:
            continue

        kept.append(s)

    return "\n".join(kept)

# emoji regex
_EMOJI_RE = re.compile(
    r"[\U0001F1E6-\U0001F1FF"
    r"\U0001F300-\U0001FAFF"
    r"\U00002700-\U000027BF"
    r"\u2600-\u26FF"
    r"]",
    flags=re.UNICODE,
)

_DISALLOWED_CHARS_RE = re.compile(r"[^\w\s<>\-\'\".,!?():;/#]")
_URL_TRAIL_CHARS = ")]}>,.!?\"'"

HashtagMode = Literal["keep_text", "placeholder", "remove"]
EmojiMode = Literal["keep", "placeholder", "remove"]
UrlMode = Literal["placeholder", "remove", "keep", "domain"]
MentionMode = Literal["keep", "placeholder", "remove"]

_DEFAULT_SLANG: Dict[str, str] = {
    "fr": "for real",
    "idk": "i don't know",
    "smh": "shaking my head",
    "lol": "laughing out loud",
}

_TRANSLATE_PUNCT = str.maketrans({
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2014": "-",
    "\u2013": "-",
    "\u2026": "...",
    "\u00a0": " ",
})

def _is_nan(x: object) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and math.isnan(x):
        return True
    if _HAS_PANDAS:
        try:
            return bool(pd.isna(x))
        except Exception:
            return False
    return False

def _safe_domain(url: str) -> str:
    """
    Return a best-effort domain token from a URL-like string.
    Keeps it simple (netloc without port), no extra deps.
    """
    u = url
    if u.lower().startswith("www."):
        u = "http://" + u
    parsed = urlparse(u)
    host = parsed.netloc or parsed.path
    host = host.split("/")[0].split(":")[0]
    host = host.lower().strip(".")
    return host or "<URL>"

def _replace_urls(text: str, url_mode: UrlMode, url_placeholder: str) -> str:
    """
    Replace URLs while preserving trailing punctuation like ')', '.'.
    url_mode:
      - placeholder: <URL>
      - remove: delete URL entirely (keep trailing punct)
      - keep: keep URL as-is
      - domain: replace with domain token (e.g., nytimes.com)
    """

    if url_mode == "keep":
        return text

    def repl(m: re.Match) -> str:
        url = m.group(0)
        stripped = url.rstrip(_URL_TRAIL_CHARS)
        trail = url[len(stripped):]

        if url_mode == "remove":
            core = " "
        elif url_mode == "domain":
            core = f" {_safe_domain(stripped)} "
        else:
            core = f" {url_placeholder} "

        if trail:
            return f"{core}{trail} "
        return core

    return _URL_RE.sub(repl, text)

@lru_cache(maxsize=64)
def _slang_pattern(keys: tuple[str, ...]) -> re.Pattern:
    if not keys:
        return re.compile(r"$^")
    return re.compile(r"\b(" + "|".join(map(re.escape, keys)) + r")\b", re.IGNORECASE)

def expand_slang(text: str, slang_dict: Optional[Dict[str, str]] = None) -> str:
    slang_raw = slang_dict or _DEFAULT_SLANG
    if not slang_raw:
        return text

    slang = {str(k).lower(): str(v) for k, v in slang_raw.items()}
    pattern = _slang_pattern(tuple(sorted(slang.keys())))

    def repl(m: re.Match) -> str:
        return slang.get(m.group(1).lower(), m.group(1))

    return pattern.sub(repl, text)

def _normalizer(
    text: Union[str, object],
    *,
    lowercase: bool = True,
    normalize_unicode_punct: bool = True,
    url_mode: UrlMode = "placeholder",
    url_placeholder: str = "<URL>",
    mention_mode: MentionMode = "keep",
    mention_placeholder: str = "<USER>",
    hashtag_mode: HashtagMode = "keep_text",
    emoji_mode: EmojiMode = "remove",
    remove_special_chars: bool = True,
    collapse_whitespace: bool = True,
    dedupe_punct: bool = True,
    strip_trailing_punct: bool = False,
    do_slang_expand: bool = False,
    slang_dict: Optional[Dict[str, str]] = None,
    strip_boilerplate: bool = False,
) -> Optional[str]:
    """
    Best-practice defaults for claim/article text:
      - normalize_unicode_punct=True (fixes smart quotes/dashes/ellipsis)
      - url_mode="placeholder"
      - mention_mode="keep" (claims/articles rarely need @ -> <USER>)
      - emoji_mode="remove"
      - keep basic punctuation (.,!? etc.) unless strip_trailing_punct=True
    """
    if text is None or _is_nan(text):
        return None

    s = str(text)

    if strip_boilerplate:
        s = strip_boilerplate_heuristic(s)

    if normalize_unicode_punct:
        s = s.translate(_TRANSLATE_PUNCT)

    if lowercase:
        s = s.lower()

    # urls
    s = _replace_urls(s, url_mode=url_mode, url_placeholder=url_placeholder)

    # mentions
    if mention_mode == "placeholder":
        s = _MENTION_RE.sub(f" {mention_placeholder} ", s)
    elif mention_mode == "remove":
        s = _MENTION_RE.sub(" ", s)

    # hashtags
    if hashtag_mode == "keep_text":
        s = _HASHTAG_RE.sub(r"\1", s)
    elif hashtag_mode == "placeholder":
        s = _HASHTAG_RE.sub(" <HASHTAG> ", s)
    elif hashtag_mode == "remove":
        s = _HASHTAG_RE.sub(" ", s)

    # emojis
    if emoji_mode == "placeholder":
        s = _EMOJI_RE.sub(" <EMOJI> ", s)
    elif emoji_mode == "remove":
        s = _EMOJI_RE.sub(" ", s)

    if do_slang_expand:
        s = expand_slang(s, slang_dict=slang_dict)

    if remove_special_chars:
        if emoji_mode == "keep":
            emojis: list[str] = []

            def _emo_keep(m: re.Match) -> str:
                emojis.append(m.group(0))
                return f" <EMOJI_{len(emojis) - 1}> "

            s = _EMOJI_RE.sub(_emo_keep, s)
            s = _DISALLOWED_CHARS_RE.sub(" ", s)
            for i, e in enumerate(emojis):
                s = s.replace(f"<EMOJI_{i}>", e)
        else:
            s = _DISALLOWED_CHARS_RE.sub(" ", s)

    if dedupe_punct:
        s = _REPEAT_PUNCT_RE.sub(r"\1", s)

    if collapse_whitespace:
        s = _WHITESPACE_RE.sub(" ", s).strip()

    if strip_trailing_punct and s:
        s = s.rstrip(".,!?;:")

    return s if s else None

def normalize_text(
    text: Union[str, object],
    *,
    strip_boilerplate: bool = False,
) -> Optional[str]:
    return _normalizer(
        text,
        lowercase=True,
        normalize_unicode_punct=True,
        url_mode="placeholder",
        mention_mode="keep",
        hashtag_mode="keep_text",
        emoji_mode="remove",
        remove_special_chars=True,
        dedupe_punct=True,
        collapse_whitespace=True,
        strip_trailing_punct=False,
        do_slang_expand=False,
        strip_boilerplate=strip_boilerplate,
    )