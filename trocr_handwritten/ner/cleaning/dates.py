"""French date parser for 19th-century civil registry acts.

Parses declaration_date first (fully specified), then resolves event dates
using the declaration_date as reference for relative terms ('dernier', 'courant').
"""

import re
from datetime import date
from typing import Optional

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_FRENCH_DAYS = {
    "premier": 1,
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
    "trente-un": 31,
}

_FRENCH_MONTHS = {
    "janvier": 1,
    "février": 2,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "août": 8,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "décembre": 12,
    "decembre": 12,
}

_FRENCH_YEAR_UNITS = {
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
}

_FRENCH_TENS = {
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
    "cinquante-six": 56,
    "cinquante six": 56,
    "cinquante-sept": 57,
    "cinquante sept": 57,
    "cinquante-huit": 58,
    "cinquante huit": 58,
}

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[,\.\!\?;:]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_french_day(text: str) -> Optional[int]:
    """Extract day number from normalised French text fragment."""
    text = _normalise(text)
    # Try multi-word matches first (longest first)
    for phrase in sorted(_FRENCH_DAYS, key=len, reverse=True):
        if text.startswith(phrase):
            return _FRENCH_DAYS[phrase]
    # Numeric fallback
    m = re.match(r"^(\d{1,2})", text)
    if m:
        return int(m.group(1))
    return None


def _parse_french_month(text: str) -> Optional[int]:
    """Return month number from a normalised French month name."""
    text = _normalise(text)
    for name, num in _FRENCH_MONTHS.items():
        if name in text:
            return num
    return None


def _parse_french_year(text: str) -> Optional[int]:
    """Parse 'mil huit cent quarante deux' style year from normalised text."""
    text = _normalise(text)
    # Numeric year
    m = re.search(r"\b(18\d{2})\b", text)
    if m:
        return int(m.group(1))
    # 'mil huit cent ...'
    if "mil huit cent" not in text and "mil-huit-cent" not in text:
        return None
    suffix = re.sub(r".*mil.{0,5}huit.{0,5}cent\s*", "", text).strip()
    if not suffix or suffix in ("", "s"):
        return 1800
    for phrase in sorted(_FRENCH_TENS, key=len, reverse=True):
        if suffix.startswith(phrase):
            return 1800 + _FRENCH_TENS[phrase]
    return None


# ---------------------------------------------------------------------------
# Full declaration date parser (always fully specified)
# ---------------------------------------------------------------------------


def parse_declaration_date(text: str) -> Optional[date]:
    """Parse a fully-specified French declaration date.

    Handles: 'Lundi quatre du mois de Janvier mil huit cent quarante un'
    Returns a date object or None if unparseable.
    """
    if not text or text.strip().lower() in ("null", "none", ""):
        return None

    norm = _normalise(text)

    # Strip leading weekday
    norm = re.sub(
        r"^(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\s+", "", norm
    )
    # Strip 'ce jour' / 'aujourd'hui'
    norm = re.sub(r"^(ce jour|aujourd.?hui)\s*", "", norm)

    day = _parse_french_day(norm)

    month = _parse_french_month(norm)
    year = _parse_french_year(norm)

    if day and month and year:
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Event date parser (may contain 'dernier' / 'courant')
# ---------------------------------------------------------------------------


def parse_event_date(text: str, declaration_date: Optional[date]) -> Optional[str]:
    """Parse an event date, resolving 'dernier'/'courant' via declaration_date.

    Returns ISO string 'YYYY-MM-DD', 'YYYY-MM', or 'YYYY', or None.
    """
    if not text or text.strip().lower() in ("null", "none", "", "ce jour"):
        if text and "ce jour" in text.lower() and declaration_date:
            return declaration_date.isoformat()
        return None

    norm = _normalise(text)

    # Strip leading weekday
    norm = re.sub(
        r"^(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\s+", "", norm
    )

    has_dernier = "dernier" in norm or "der " in norm
    has_courant = "courant" in norm or "cour " in norm

    day = _parse_french_day(norm)
    month = _parse_french_month(norm)
    year = _parse_french_year(norm)

    # Resolve relative month references using declaration_date
    if declaration_date and not year:
        if has_dernier:
            ref_month = declaration_date.month - 1 or 12
            year = (
                declaration_date.year - 1
                if declaration_date.month == 1
                else declaration_date.year
            )
            if not month:
                month = ref_month
        elif has_courant:
            year = declaration_date.year
            if not month:
                month = declaration_date.month
        else:
            year = declaration_date.year

    if day and month and year:
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            pass
    if month and year:
        return f"{year:04d}-{month:02d}"
    if year:
        return str(year)
    return None


# ---------------------------------------------------------------------------
# Convenience: parse both dates for a single act row
# ---------------------------------------------------------------------------


def parse_act_dates(
    declaration_date_raw: Optional[str],
    event_date_raw: Optional[str],
    year_registry: Optional[int] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Return (declaration_date_iso, event_date_iso).

    Args:
        declaration_date_raw: Raw declaration date string from LLM.
        event_date_raw: Raw event date string (birth/death/marriage date).
        year_registry: Year extracted from act_id, used as fallback.
    """
    decl = parse_declaration_date(declaration_date_raw)

    # If declaration_date failed but year_registry is known, use Jan 1 as proxy
    if decl is None and year_registry:
        decl = date(year_registry, 1, 1)
        decl_iso = None
    else:
        decl_iso = decl.isoformat() if decl else None

    event_iso = parse_event_date(event_date_raw, decl)
    return decl_iso, event_iso
