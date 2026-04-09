"""French age parser for 19th-century civil registry acts."""

import re
from typing import Optional

_FRENCH_NUMBERS = {
    "un": 1,
    "une": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
    "onze": 11,
    "douze": 12,
    "treize": 13,
    "quatorze": 14,
    "quinze": 15,
    "seize": 16,
    "dix-sept": 17,
    "dix sept": 17,
    "dix-huit": 18,
    "dix huit": 18,
    "dix-neuf": 19,
    "dix neuf": 19,
    "vingt": 20,
    "vingt et un": 21,
    "vingt-et-un": 21,
    "vingt-un": 21,
    "vingt-deux": 22,
    "vingt deux": 22,
    "vingt-trois": 23,
    "vingt trois": 23,
    "vingt-quatre": 24,
    "vingt quatre": 24,
    "vingt-cinq": 25,
    "vingt cinq": 25,
    "vingt-six": 26,
    "vingt six": 26,
    "vingt-sept": 27,
    "vingt sept": 27,
    "vingt-huit": 28,
    "vingt huit": 28,
    "vingt-neuf": 29,
    "vingt neuf": 29,
    "trente": 30,
    "trente et un": 31,
    "trente-et-un": 31,
    "trente-deux": 32,
    "trente deux": 32,
    "trente-trois": 33,
    "trente trois": 33,
    "trente-quatre": 34,
    "trente quatre": 34,
    "trente-cinq": 35,
    "trente cinq": 35,
    "trente-six": 36,
    "trente six": 36,
    "trente-sept": 37,
    "trente sept": 37,
    "trente-huit": 38,
    "trente huit": 38,
    "trente-neuf": 39,
    "trente neuf": 39,
    "quarante": 40,
    "quarante et un": 41,
    "quarante-et-un": 41,
    "quarante-deux": 42,
    "quarante deux": 42,
    "quarante-trois": 43,
    "quarante trois": 43,
    "quarante-quatre": 44,
    "quarante quatre": 44,
    "quarante-cinq": 45,
    "quarante cinq": 45,
    "quarante-six": 46,
    "quarante six": 46,
    "quarante-sept": 47,
    "quarante sept": 47,
    "quarante-huit": 48,
    "quarante huit": 48,
    "quarante-neuf": 49,
    "quarante neuf": 49,
    "cinquante": 50,
    "cinquante et un": 51,
    "cinquante-et-un": 51,
    "cinquante-deux": 52,
    "cinquante deux": 52,
    "cinquante-trois": 53,
    "cinquante trois": 53,
    "cinquante-quatre": 54,
    "cinquante quatre": 54,
    "cinquante-cinq": 55,
    "cinquante cinq": 55,
    "soixante": 60,
    "soixante et un": 61,
    "soixante-et-un": 61,
    "soixante-dix": 70,
    "soixante dix": 70,
    "quatre-vingts": 80,
    "quatre vingts": 80,
    "quatre-vingt": 80,
    "quatre-vingt-dix": 90,
    "quatre vingt dix": 90,
    "cent": 100,
}

_NOISE_PATTERN = re.compile(
    r"\b(environ|à peu près|a peu pres|environ|age de|âgé de|agé de|âgé|age|ans?)\b",
    re.IGNORECASE,
)


def parse_age(text: Optional[str]) -> Optional[int]:
    """Parse a French age string to an integer.

    Examples:
        'Vingt huit ans' -> 28
        'environ 40 ans' -> 40
        '35' -> 35
    """
    if not text or str(text).strip().lower() in ("null", "none", ""):
        return None

    norm = text.lower().strip()
    norm = _NOISE_PATTERN.sub(" ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()

    # Numeric
    m = re.search(r"\b(\d+)\b", norm)
    if m:
        val = int(m.group(1))
        if 0 < val < 120:
            return val

    # French words — longest match first
    for phrase in sorted(_FRENCH_NUMBERS, key=len, reverse=True):
        if phrase in norm:
            return _FRENCH_NUMBERS[phrase]

    return None


def parse_age_validated(text: Optional[str], min_age: int = 0) -> Optional[int]:
    """Parse age and return None if below min_age threshold."""
    age = parse_age(text)
    if age is not None and age < min_age:
        return None
    return age
