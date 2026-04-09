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
    "dix huis": 18,
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
    "trente un": 31,
}


_FRENCH_MONTHS = {
    "janvier": 1,
    "février": 2,
    "fevrier": 2,
    "fevrai": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "août": 8,
    "aout": 8,
    "septembre": 9,
    "7bre": 9,
    "octobre": 10,
    "8bre": 10,
    "novembre": 11,
    "9bre": 11,
    "décembre": 12,
    "decembre": 12,
    "xbre": 12,
    "10bre": 12,
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
    "quarante un": 41,
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
    "cinquante un": 51,
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
    """Lowercase, collapse whitespace, strip punctuation and OCR noise."""
    text = text.lower().strip()
    # Remove bracketed illegible tokens and uncertainty markers
    text = re.sub(r"\[illisible[^\]]*\]", " ", text)
    text = re.sub(r"[\[\]?]", "", text)
    # Strip time-of-day phrases
    text = re.sub(r"à\s+\w+\s+heures.*$", "", text)
    text = re.sub(r"[,\.\!\?;:()\[\]]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_french_day(text: str) -> Optional[int]:
    """Extract day number from a short text (no month context)."""
    text = _normalise(text)
    for phrase in sorted(_FRENCH_DAYS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            return _FRENCH_DAYS[phrase]
    m = re.search(r"\b(\d{1,2})\b", text)
    if m:
        return int(m.group(1))
    return None


def _find_month_position(norm: str) -> tuple[Optional[int], Optional[int]]:
    """Return (month_number, start_index_of_month_token) for the first month found."""
    for name, num in sorted(
        _FRENCH_MONTHS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        m = re.search(r"\b" + re.escape(name) + r"\b", norm)
        if m:
            return num, m.start()
    return None, None


def _parse_day_from_context(norm: str, month_start: int) -> Optional[int]:
    """Parse day from tokens before the month, skipping 'du mois de/d'' fillers."""
    before = norm[:month_start].strip()
    # Strip trailing filler between day and month: "du présent mois", "du courant mois", "du mois de/d'", "de ce mois"
    before = re.sub(r"\s+d[eu]\s+\w+\s+mois\s*$", "", before).strip()
    before = re.sub(r"\s+du\s+mois\s+d[e']?\s*$", "", before).strip()
    before = re.sub(r"\s+de\s+ce\s+mois\s*$", "", before).strip()
    tokens = before.split()
    window = " ".join(tokens[-4:]) if tokens else ""
    for phrase in sorted(_FRENCH_DAYS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(phrase) + r"\b", window):
            return _FRENCH_DAYS[phrase]
    digit = re.search(r"\b(\d{1,2})\b", window)
    if digit:
        return int(digit.group(1))
    return None


def _parse_french_month(text: str) -> Optional[int]:
    """Return month number from a normalised French month name, with fuzzy fallback."""
    norm = _normalise(text)
    for name, num in _FRENCH_MONTHS.items():
        if name in norm:
            return num
    # Fuzzy fallback: match any token in the text against known month names
    try:
        from rapidfuzz import process, fuzz

        tokens = norm.split()
        for token in tokens:
            if len(token) < 3:
                continue
            match = process.extractOne(
                token, list(_FRENCH_MONTHS.keys()), scorer=fuzz.ratio, score_cutoff=80
            )
            if match:
                return _FRENCH_MONTHS[match[0]]
    except ImportError:
        pass
    return None


def _parse_french_year(text: str) -> Optional[int]:
    """Parse 'mil huit cent quarante deux' style year from normalised text."""
    text = _normalise(text)
    # Numeric year
    m = re.search(r"\b(18\d{2})\b", text)
    if m:
        return int(m.group(1))
    # 'mil huit cent ...' — allow OCR/spelling variants: cens, huis, huip
    if not re.search(r"mil.{0,6}hui[tsp].{0,6}cen[ts]", text):
        return None
    suffix = re.sub(r".*mil.{0,6}hui[tsp].{0,6}cen[ts]\s*", "", text).strip()
    if not suffix or suffix in ("", "s"):
        return 1800
    for phrase in sorted(_FRENCH_TENS, key=len, reverse=True):
        if suffix.startswith(phrase):
            return 1800 + _FRENCH_TENS[phrase]
    return None


# ---------------------------------------------------------------------------
# Full declaration date parser (always fully specified)
# ---------------------------------------------------------------------------


def parse_declaration_date(
    text: str, year_registry: Optional[int] = None
) -> Optional[date]:
    """Parse a fully-specified French declaration date.

    Handles: 'Lundi quatre du mois de Janvier mil huit cent quarante un'
    Uses year_registry as fallback when year is absent from text.
    Returns a date object or None if unparseable.
    """
    import calendar

    if not text or text.strip().lower() in ("null", "none", ""):
        return None

    norm = _normalise(text)

    # Strip leading apostrophe/l'an / l an / le
    norm = re.sub(r"^['\u2019]?\s*l.?an\s+", "", norm)
    norm = re.sub(r"^le\s+", "", norm)

    # "dernier jour du mois de [month]"
    m_dernier = re.match(r"dernier jour du mois de (\w+)", norm)
    if m_dernier:
        month = _parse_french_month(m_dernier.group(1))
        year = _parse_french_year(norm) or year_registry
        if month and year:
            last_day = calendar.monthrange(year, month)[1]
            try:
                return date(year, month, last_day)
            except ValueError:
                return None
        return None

    year = _parse_french_year(norm) or year_registry

    month, month_start = _find_month_position(norm)
    if month_start is not None:
        day = _parse_day_from_context(norm, month_start)
    else:
        month = _parse_french_month(norm)
        day = _parse_french_day(norm)

    if day and month and year:
        try:
            return date(year, month, day)
        except ValueError:
            return None
    # month+year only (no day): use day=1 as proxy
    if month and year:
        return date(year, month, 1)
    return None


# ---------------------------------------------------------------------------
# Event date parser (may contain 'dernier' / 'courant')
# ---------------------------------------------------------------------------


def parse_event_date(text: str, declaration_date: Optional[date]) -> Optional[str]:
    """Parse an event date, resolving 'dernier'/'courant' via declaration_date.

    Returns ISO string 'YYYY-MM-DD', 'YYYY-MM', or 'YYYY', or None.
    """
    if not text or text.strip().lower() in ("null", "none", ""):
        return None

    norm = _normalise(text)

    # Relative-day shortcuts
    if re.search(r"\b(ce jour|aujourd.?hui)\b", norm) and declaration_date:
        return declaration_date.isoformat()
    if norm.startswith("hier") and declaration_date:
        from datetime import timedelta

        return (declaration_date - timedelta(days=1)).isoformat()
    if norm.startswith("avant hier") and declaration_date:
        from datetime import timedelta

        return (declaration_date - timedelta(days=2)).isoformat()

    has_dernier = "dernier" in norm or "der " in norm
    has_courant = (
        "courant" in norm or "cour " in norm or "présent" in norm or "present" in norm
    )

    year = _parse_french_year(norm)

    month, month_start = _find_month_position(norm)
    if month_start is not None:
        day = _parse_day_from_context(norm, month_start)
    else:
        month = _parse_french_month(norm)
        day = _parse_french_day(norm)

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
        return f"{year:04d}-{month:02d}-01"
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
    decl = parse_declaration_date(declaration_date_raw, year_registry=year_registry)
    decl_iso = decl.isoformat() if decl else None

    event_iso = parse_event_date(event_date_raw, decl)
    return decl_iso, event_iso
