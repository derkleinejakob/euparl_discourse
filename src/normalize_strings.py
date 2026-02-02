import unicodedata
import re
from typing import Optional
import pandas as pd

## define some helper functions

# 00 Normalize names
def normalize_name(x: object) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None

    # Unicode normalize (handles different codepoint compositions)
    s = unicodedata.normalize("NFKC", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)

    # Optional: make case-insensitive comparisons safer across languages
    # (keeps Greek/Latin letters as-is; just normalizes case)
    s = s.casefold()

    return s

# 01_Normalize the strings (lower-casing, removing accents/diacritics and stripping extra whitespace)
def normalize_str(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s

#print(normalize_str("ÖVP"))
#print(normalize_str("Parti québécois"))
#print(normalize_str(" ÖVP "))

# helper function for lists (creates a normalized set)
#def normalize_list(lst):
#    return {normalize_str(x) for x in lst if isinstance(x, str)}
#
#print(normalize_list(["ÖVP", "Parti québécois", " ÖVP "]))

#
#
#
#
#
#
#
#
#


"""
Strict name normalization for matching.

Goal: reduce a personal name to a very basic form so that small differences
(whitespace, punctuation, case, accents/diacritics) disappear.

Example:
  "Jiří POSPÍŠIL" and "Jíri Pospísíl" -> "jiripospisil"
"""

_ALNUM_RE = re.compile(r"[^0-9a-z]+")  # after casefold+ascii translit we keep a-z0-9

def _strip_diacritics(s: str) -> str:
    # NFKD splits letters+diacritics; we drop combining marks
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )

def normalize_name_strict(name: object, *, transliterate: bool = True) -> Optional[str]:
    """
    Return a strict normalized key for a name, or None if empty.

    Steps:
      1) NFKC normalize
      2) casefold
      3) transliterate to ASCII Latin (optional, requires Unidecode)
      4) strip diacritics
      5) keep only [a-z0-9], remove everything else (spaces, punctuation, hyphens, etc.)
    """
    if name is None:
        return None

    s = str(name).strip()
    if not s:
        return None

    # 1) Unicode canonicalization
    s = unicodedata.normalize("NFKC", s)

    # 2) Case-insensitive across languages
    s = s.casefold()

    # 3) Transliterate Greek/Cyrillic/etc. to Latin if possible
    if transliterate:
        try:
            from unidecode import unidecode  # pip install Unidecode
            s = unidecode(s)
        except Exception:
            # If Unidecode isn't available, we continue without transliteration
            pass

    # 4) Remove accents/diacritics
    s = _strip_diacritics(s)

    # 5) Keep only letters/digits; this also removes whitespace between names
    s = _ALNUM_RE.sub("", s)

    return s or None



#tests = [
#    "Nótis Mariás",
#    "Jiří POSPÍŠIL",
#    "Jíri Pospísíl",
#    " Jean-Luc  Mélenchon ",
#    "O’Connor",
#    "Κυριάκος Μητσοτάκης",  # needs Unidecode to transliterate well
#    "Бойко Борисов",        # needs Unidecode to transliterate well
#]
#for t in tests:
#    print(f"{t!r} -> {normalize_name_strict(t)}")