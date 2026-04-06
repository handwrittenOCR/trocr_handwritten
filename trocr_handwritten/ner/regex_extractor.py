"""Regex-based NER extraction for 19th-century Guadeloupe civil registry acts.

Patterns are tuned for the formulaic legal French of slave registries (1841-1848),
with tolerance for OCR transcription variations.
"""

import logging
import re
from typing import List, Optional, Tuple

from trocr_handwritten.ner.schemas import (
    ActRecord,
    BirthActEntity,
    DeathActEntity,
    NERResult,
    PersonInfo,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# French number conversion (spelled-out -> integer)
# ---------------------------------------------------------------------------

_UNITS = {
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
    "trente": 30,
    "quarante": 40,
    "cinquante": 50,
    "soixante": 60,
}


def french_number_to_int(text: str) -> Optional[int]:
    """Convert a French spelled-out number to an integer (0-120 range).

    Examples:
        'vingt sept' -> 27
        'soixante dix' -> 70
        'quatre vingt seize' -> 96
        'cent' -> 100
        'trois' -> 3
    """
    if not text:
        return None
    text = text.strip().lower().replace("-", " ").replace("  ", " ")

    # Direct digit match
    if text.isdigit():
        return int(text)

    # "cent" / "cent X"
    if text == "cent":
        return 100
    if text.startswith("cent "):
        rest = french_number_to_int(text[5:])
        return 100 + rest if rest is not None else None

    # "quatre vingt(s)" = 80
    qv_match = re.match(r"quatre\s+vingts?(?:\s+(.+))?$", text)
    if qv_match:
        rest = qv_match.group(1)
        if not rest:
            return 80
        rest_val = french_number_to_int(rest)
        return 80 + rest_val if rest_val is not None else None

    # "soixante dix/onze/douze..." = 70+
    sx_match = re.match(r"soixante\s+(.+)$", text)
    if sx_match:
        rest = sx_match.group(1)
        rest_val = french_number_to_int(rest)
        if rest_val is not None and rest_val >= 10:
            return 60 + rest_val
        # soixante + unit (e.g. soixante cinq = 65)
        if rest_val is not None:
            return 60 + rest_val

    # Tens + units: "trente deux", "quarante cinq"
    for tens_word, tens_val in [
        ("vingt", 20),
        ("trente", 30),
        ("quarante", 40),
        ("cinquante", 50),
    ]:
        if text.startswith(tens_word + " "):
            rest = text[len(tens_word) + 1 :]
            if rest == "et un" or rest == "et une":
                return tens_val + 1
            rest_val = french_number_to_int(rest)
            if rest_val is not None:
                return tens_val + rest_val

        if text == tens_word:
            return tens_val

    # Direct lookup
    if text in _UNITS:
        return _UNITS[text]

    return None


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Person designation: captures racial/status term, optional "nommé(e)", name, and age
# Matches: "la negresse angelle agée de vingt sept ans"
#          "le petit nègre Caraïse agé de un an"
#          "son nègre St Pierre dit Coquin, agé de trente trois ans"
#          "la rouge ANNE N° 1 agée de Vingt sept ans"
_PERSON_DEATH_PATTERN = re.compile(
    r"(?:l[ea]|sa|son)\s+"
    r"(?:petit[es]?\s+)?"
    r"(n[éeèi]gr(?:esse|illon(?:ne)?|ett?e|iss?e|e)|rouge)\s+"
    r"(?:nomm[ée]+\s+)?"
    r"(.+?)"
    r"\s*,?\s*[aâàé]g[ée]+[es]?\s+de\s+"
    r"(.+?)\s+ans?",
    re.IGNORECASE | re.DOTALL,
)

# Occupation: appears after "X ans, [OCCUPATION], immatriculé/appartenant"
# Known occupations in these registries
_OCCUPATION_WORDS = (
    "cultivateur",
    "cultivatrice",
    "domestique",
    "sans profession",
    "ouvrier",
    "ouvrière",
    "marchande",
    "marchand",
    "infirme",
    "cuisinière",
    "cuisinier",
    "blanchisseuse",
    "charpentier",
    "maçon",
    "charretier",
    "tonnelier",
    "journalier",
    "journalière",
)

_OCCUPATION_PATTERN = re.compile(
    r"ans\s*,?\s*(" + "|".join(re.escape(w) for w in _OCCUPATION_WORDS) + r")\s*,?\s*"
    r"(?:[eéi]mma|appar|port)",
    re.IGNORECASE,
)

# Registration: "immatriculé(e) Registre D N° 3328" / "Regestre A N° 17"
_REGISTRATION_PATTERN = re.compile(
    r"imma[\-\s]?\s*tricul[ée]+[es]?\s+"
    r"[Rr]eg[ie]?stre\s+([A-Z])[\s.]*"
    r"[Nn][°ᵒo˚]?\s*(\d+)",
    re.IGNORECASE,
)

# Habitation: anchor on the word "habitation", capture what follows until a delimiter
_HABITATION_PATTERN = re.compile(
    r"habi[\-\s]?\s*tation\s+"
    r"(.+?)"
    r"(?:\s+de\s+cette|\s+appar[\-\s]?\s*tenant|\s*,|\s+[sl](?:a|on|es)\s+n[éeèi]gr|\s+le\s+(?:n[éeèi]gr|petit|rouge)|\s+la\s+(?:n[éeèi]gr|petit|rouge|capresse)|\s+[sl](?:a|on|es)\s+(?:petit|rouge)|\s+n[éeèi]gr(?:esse|e)|\s+immatricul|\s+est\s+|\s+y\s+est|\s+port[ée])",
    re.IGNORECASE | re.DOTALL,
)

# Owner pattern 1: "appartenant à/au [TITLE] [NAME]"
# Captures everything after "appartenant à/au" until a delimiter
_OWNER_APPARTENANT_PATTERN = re.compile(
    r"appar[\-\s]?\s*tenant\s+(?:au[x]?\s+|[àa]\s+)"
    r"(.+?)"
    r"(?:\s*,|\s+de\s+cette|\s+immatricul|\s+le\s+n[éeèi]gr|\s+la\s+n[éeèi]gr|\s+le\s+(?:petit|rouge)|\s+la\s+(?:petit|rouge|capresse)|\s+et\s+dans|\s+y\s+est|\s+est\s+d[ée]c[ée]d|\s+est\s+accouch|\s+et\s+a\s+sign)",
    re.IGNORECASE | re.DOTALL,
)

# Owner pattern 2: "porté(e) sur les recensements de/des [NAME]"
_OWNER_RECENS_PATTERN = re.compile(
    r"port[ée]+[es]?\s+sur\s+les\s+recense?ments?\s+(?:de\s+|des\s+)"
    r"(.+?)"
    r"(?:\s*,|\s+y\s+est|\s+immatricul|\s+est\s+d[ée]c[ée]d)",
    re.IGNORECASE | re.DOTALL,
)

# "son habitation" / "sa négresse" -> owner is the declarant
_POSSESSIVE_PATTERN = re.compile(
    r"(?:son\s+habi[\-\s]?\s*tation|sa\s+n[éeèi]gr)",
    re.IGNORECASE,
)

# Declaration date: "L'An Mil huit cent quarante deux et le lundi trois du mois de Janvier"
_DECLARATION_DATE_PATTERN = re.compile(
    r"(?:le\s+)?"
    r"(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\s+"
    r"(.+?)"
    r"\s+(?:du\s+mois\s+d[e'\u2019]\s*|de\s+)"
    r"([a-zéèàûü]+)",
    re.IGNORECASE,
)

# Declaration time: "à sept heures du matin" / "à trois heures de l'après midi"
_DECLARATION_TIME_PATTERN = re.compile(
    r"[àa]\s+(.+?heures?(?:\s+et\s+demi[e]?)?)\s+"
    r"du\s+(matin|soir)|de\s+l['\u2019]apr[èe]s[\s\-]?midi",
    re.IGNORECASE,
)

# Declarant: "Est comparu le Sieur [NAME] agé de [AGE] ans, [OCCUPATION]"
# Name can span line breaks. Occupation ends at "domicilié" or "en la" or "Lequel"
_DECLARANT_PATTERN = re.compile(
    r"[Ee][Ss][Tt]\s+[Cc]omparu\s+le\s+[Ss]ieur\s+"
    r"(.+?)\s+"
    r"[aâàé]g[ée]+\s+de\s+(.+?)\s+ans\s*,?\s*"
    r"(habitant[e]?\s*[,\s]?\s*(?:propri[ée]taire|g[ée]r[ea]nt|cultivateur|g[ée]reur)?[^,;]*?)"
    r"(?:\s*,?\s*domicili[ée]|\s+en\s+la\s+(?:dite\s+)?commune|\s+de\s+cette\s+commune|\s*[,;]\s*[Ll]equel)",
    re.IGNORECASE | re.DOTALL,
)

# Birth: child sex and name
# Variants: "enfant noir du sexe féminin qui a eu nom X"
#           "enfant né du sexe feminin qui a eu nom X"
#           "enfant mâle qui a eu nom X"
#           "enfant noir, qui a eu nom X"
#           "enfant ... qui a reçu nom X"
_BIRTH_CHILD_PATTERN = re.compile(
    r"accouch[ée]+e?\s+d['\u2019]un\s+enfant\s+"
    r"(?:noir[e]?\s*,?\s*|rouge\s*,?\s*|n[ée]\s+)?"
    r"(?:du\s+sexe\s+(masculin|f[ée]minin)\s*,?\s*|"
    r"(m[aâ]le|femelle)\s*,?\s*)?"
    r"qui\s+a\s+(?:eu|re[çc]u)\s+nom\s+"
    r"(.+?)"
    r"\s*,?\s*[eéi]mma",
    re.IGNORECASE | re.DOTALL,
)

# Birth: child registration (appears after the child name)
_BIRTH_CHILD_REG_PATTERN = re.compile(
    r"qui\s+a\s+(?:eu|re[çc]u)\s+nom\s+.+?"
    r"[eéi]mma[\-\s]?\s*tricul[ée]+[es]?\s+"
    r"[Rr]eg[ie]?stre\s+([A-Z])[\s.]*"
    r"[Nn][°ᵒo˚]?\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)

# Birth: mother is the person who "est accouchée" — reuse person pattern
# but specifically look for the person right before "est accouchée"
_BIRTH_MOTHER_PATTERN = re.compile(
    r"(?:l[ea]|sa|son)\s+"
    r"(?:petit[es]?\s+)?"
    r"(n[éeèi]gr(?:esse|illon(?:ne)?|ett?e|iss?e|e)|rouge)\s+"
    r"(?:nomm[ée]+\s+)?"
    r"(.+?)"
    r"\s*,?\s*[aâàé]g[ée]+[es]?\s+de\s+"
    r"(.+?)\s+ans?\s*,?\s*"
    r"imma[\-\s]?\s*tricul[ée]+[es]?\s+"
    r"[Rr]eg[ie]stre\s+([A-Z]?)\s*"
    r"[Nn]?[°ᵒo˚]?\s*(\d*)"
    r".*?(?:y\s+)?est\s+accouch[ée]+",
    re.IGNORECASE | re.DOTALL,
)

# Event date/time: "que ce matin à une heure" / "que hier soir à sept heures"
_EVENT_TIME_PATTERN = re.compile(
    r"(?:que\s+)"
    r"(ce\s+matin|hier\s+(?:soir|matin)?|"
    r"(?:le\s+)?(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\s+.+?)"
    r"\s+[àa]\s+(.+?heures?(?:\s+et\s+demi[e]?)?)",
    re.IGNORECASE,
)

# Death indicator: "y est décédé(e)"
_DEATH_INDICATOR = re.compile(r"\bd[ée]c[ée]d[ée]|\bd[ée]c[ée]s\b", re.IGNORECASE)

# Birth indicator: "est accouchée"
_BIRTH_INDICATOR = re.compile(r"\baccouch[ée]+\b|\bné(e|é)\b", re.IGNORECASE)

# Officer: "Pardevant nous [NAME], Maire" or "Pardevant nous [NAME] adjoint"
_OFFICER_PATTERN = re.compile(
    r"[Pp]ardevant\s+[Nn]ous\s+"
    r"(.+?)"
    r"\s*,?\s*(?:[Mm]aire|adjoint|officier|et\s+officier)",
    re.IGNORECASE | re.DOTALL,
)

# Commune
_COMMUNE_PATTERN = re.compile(r"commune\s+des?\s+([a-zà-ü]+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _sex_from_designator(designator: str) -> str:
    """Infer sex from the racial/status designator word."""
    d = designator.lower()
    # Female forms: negresse, négrette, negritte, negrisse, nigresse, négrillonne
    if any(f in d for f in ("gresse", "grette", "gritte", "grisse", "grillonne")):
        return "femme"
    # Male forms: negre, nègre, négrillon, negrillon, nigre
    if any(f in d for f in ("gre", "grillon")):
        return "homme"
    # "rouge" is gender-neutral — need context
    return "femme"


def _extract_declarant(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract declarant name, age, occupation from text."""
    match = _DECLARANT_PATTERN.search(text)
    if not match:
        return None, None, None
    name = re.sub(r"\s+", " ", match.group(1).strip().rstrip(","))
    age_text = re.sub(r"\s+", " ", match.group(2).strip())
    occupation = re.sub(r"\s+", " ", match.group(3).strip().rstrip(",").strip())
    age_int = french_number_to_int(age_text)
    age_str = str(age_int) if age_int is not None else age_text
    return name, age_str, occupation


def _extract_declaration_date(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract declaration date and time."""
    date_match = _DECLARATION_DATE_PATTERN.search(text)
    date_str = None
    if date_match:
        day = date_match.group(1).strip()
        month = date_match.group(2).strip()
        date_str = f"{day} {month}"

    time_match = _DECLARATION_TIME_PATTERN.search(text)
    time_str = None
    if time_match:
        time_str = time_match.group(0).strip()
        # Clean up: extract just the time part
        t = re.search(r"[àa]\s+(.+)", time_str, re.IGNORECASE)
        if t:
            time_str = t.group(1).strip()

    return date_str, time_str


def _extract_habitation_and_owner(
    text: str, declarant_name: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Extract habitation name and owner name."""
    hab_match = _HABITATION_PATTERN.search(text)
    habitation = None
    if hab_match:
        raw_hab = re.sub(r"\s+", " ", hab_match.group(1).strip())
        # If the capture starts with "du sieur/sr" or a title, it's an owner not a habitation
        if re.match(
            r"^(?:du\s+(?:sieur|sr)|de\s+(?:Monsieur|Madame|Mme|Dame|Demoiselle))\b",
            raw_hab,
            re.IGNORECASE,
        ):
            habitation = None
        else:
            # Clean leading prefixes
            habitation = re.sub(
                r"^(?:dite\s+|de\s+(?:la\s+|l['\u2019])?|la\s+|le\s+|d['\u2019])",
                "",
                raw_hab,
                flags=re.IGNORECASE,
            ).strip()
            # Discard if result is clearly not a place name
            if (
                not habitation
                or len(habitation) < 2
                or re.match(
                    r"^(?:un\s+enfant|dame|demoiselle|sa\s+|cette\s+commune|[àa]\s+|le\s+\d|vingt|trente|quarante|premier|deux\s)",
                    habitation,
                    re.IGNORECASE,
                )
            ):
                habitation = None

    # Try owner patterns in order of specificity
    owner = None
    for pattern in (_OWNER_APPARTENANT_PATTERN, _OWNER_RECENS_PATTERN):
        match = pattern.search(text)
        if match:
            owner = re.sub(r"\s+", " ", match.group(1).strip())
            # Strip titles from owner name
            owner = re.sub(
                r"^(?:la\s+)?(?:Monsieur|Madame|Mme|Sieur[s]?|Dame|Demoiselle|Sr)\s+"
                r"(?:Veuve\s+|Vve\s+)?(?:de\s+)?",
                "",
                owner,
                flags=re.IGNORECASE,
            ).strip()
            if not owner:
                owner = None
            break

    # If "son habitation" / "sa négresse" -> owner is the declarant
    if owner is None and _POSSESSIVE_PATTERN.search(text) and declarant_name:
        owner = declarant_name

    return habitation, owner


def _extract_officer(text: str) -> Optional[str]:
    """Extract the name of the officier de l'état civil (mayor or adjoint)."""
    match = _OFFICER_PATTERN.search(text)
    if match:
        name = re.sub(r"\s+", " ", match.group(1).strip()).rstrip(",")
        # Filter out if it's just a title with no actual name
        if re.match(
            r"^(?:le|la|l['\u2019]|[Mm]aire|adjoint|officier|[Mm]aire\s*[&y]?|"
            r"[Aa]djoint\s*,?\s*faisant\s+fonct[io]?[no]?s?\s+de)$",
            name,
            re.IGNORECASE,
        ):
            return None
        # Strip leading "adjoint faisant fonctions de Maire" prefix to get the name
        name = re.sub(
            r"^(.+?)\s*,?\s*adjoint\s*,?\s*faisant\s+fonct[io]?[no]?s?\s+de$",
            r"\1",
            name,
            flags=re.IGNORECASE,
        ).strip()
        # Clean trailing title fragments
        name = re.sub(r"\s*[&y=]\s*$", "", name).strip()
        if len(name) < 3:
            return None
        return name
    return None


def _extract_commune(text: str) -> Optional[str]:
    """Extract commune name."""
    match = _COMMUNE_PATTERN.search(text)
    return match.group(1).strip().lower() if match else None


def _extract_registration(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract the first registration (register letter + number) from text."""
    match = _REGISTRATION_PATTERN.search(text)
    if match:
        register = match.group(1).upper() if match.group(1) else None
        number = match.group(2) if match.group(2) else None
        return register, number
    return None, None


# ---------------------------------------------------------------------------
# Death act extraction
# ---------------------------------------------------------------------------


def _extract_death(record: ActRecord) -> DeathActEntity:
    """Extract entities from a death act."""
    text = record.plein_texte_text

    # Person (the deceased)
    person = PersonInfo()
    pmatch = _PERSON_DEATH_PATTERN.search(text)
    if pmatch:
        designator = pmatch.group(1)
        raw_name = re.sub(r"\s+", " ", pmatch.group(2).strip().rstrip(","))
        age_text = re.sub(r"\s+", " ", pmatch.group(3).strip())

        person.sex = _sex_from_designator(designator)
        person.name = raw_name
        age_int = french_number_to_int(age_text)
        person.age = str(age_int) if age_int is not None else age_text

    # Occupation of the deceased
    occ_match = _OCCUPATION_PATTERN.search(text)
    if occ_match:
        person.occupation = occ_match.group(1).strip()

    # Registration of the deceased (first registration in text)
    reg_register, reg_number = _extract_registration(text)
    person.registration_register = reg_register
    person.registration_number = reg_number

    # Declarant
    decl_name, decl_age, decl_occ = _extract_declarant(text)

    # Declaration date/time
    decl_date, decl_time = _extract_declaration_date(text)

    # Habitation and owner
    habitation, owner = _extract_habitation_and_owner(text, decl_name)

    # Commune
    commune = _extract_commune(text)

    # Officer
    officer = _extract_officer(text)

    # Event date/time
    event_match = _EVENT_TIME_PATTERN.search(text)
    death_date = None
    death_time = None
    if event_match:
        death_date = event_match.group(1).strip()
        death_time = event_match.group(2).strip()

    return DeathActEntity(
        person=person,
        death_date=death_date,
        death_time=death_time,
        death_place=habitation,
        declaration_date=decl_date,
        declaration_time=decl_time,
        declarant_name=decl_name,
        declarant_age=decl_age,
        declarant_occupation=decl_occ,
        owner_name=owner,
        habitation_name=habitation,
        officer_name=officer,
        commune=commune,
    )


# ---------------------------------------------------------------------------
# Birth act extraction
# ---------------------------------------------------------------------------


def _extract_birth(record: ActRecord) -> BirthActEntity:
    """Extract entities from a birth act."""
    text = record.plein_texte_text

    # Mother: the person who "est accouchée"
    mother = PersonInfo()
    mmatch = _BIRTH_MOTHER_PATTERN.search(text)
    if mmatch:
        raw_name = re.sub(r"\s+", " ", mmatch.group(2).strip().rstrip(","))
        age_text = re.sub(r"\s+", " ", mmatch.group(3).strip())

        mother.sex = "femme"
        mother.name = raw_name
        age_int = french_number_to_int(age_text)
        mother.age = str(age_int) if age_int is not None else age_text
        if mmatch.group(4):
            mother.registration_register = mmatch.group(4).upper()
        if mmatch.group(5):
            mother.registration_number = mmatch.group(5)

    # Mother's occupation
    occ_match = _OCCUPATION_PATTERN.search(text)
    if occ_match:
        mother.occupation = occ_match.group(1).strip()

    # Child
    child = PersonInfo()
    cmatch = _BIRTH_CHILD_PATTERN.search(text)
    if cmatch:
        sex_text = cmatch.group(1) or cmatch.group(
            2
        )  # "masculin/féminin" or "mâle/femelle"
        child_name = re.sub(r"\s+", " ", cmatch.group(3).strip())
        # Clean trailing punctuation and connectors
        child_name = re.sub(r"[,;]\s*(?:et\s*)?$", "", child_name).strip()
        child.name = child_name
        if sex_text:
            sex_lower = sex_text.lower()
            if "masculin" in sex_lower or "male" in sex_lower or "mâle" in sex_lower:
                child.sex = "homme"
            else:
                child.sex = "femme"

    # Child registration (appears after "qui a eu nom")
    creg_match = _BIRTH_CHILD_REG_PATTERN.search(text)
    if creg_match:
        child.registration_register = (
            creg_match.group(1).upper() if creg_match.group(1) else None
        )
        child.registration_number = creg_match.group(2) if creg_match.group(2) else None

    # Father: rarely mentioned in slave registries
    father = PersonInfo()

    # Declarant
    decl_name, decl_age, decl_occ = _extract_declarant(text)

    # Declaration date/time
    decl_date, decl_time = _extract_declaration_date(text)

    # Habitation and owner
    habitation, owner = _extract_habitation_and_owner(text, decl_name)

    # Commune
    commune = _extract_commune(text)

    # Officer
    officer = _extract_officer(text)

    # Event date/time (birth)
    event_match = _EVENT_TIME_PATTERN.search(text)
    birth_date = None
    birth_time = None
    if event_match:
        birth_date = event_match.group(1).strip()
        birth_time = event_match.group(2).strip()

    return BirthActEntity(
        child=child,
        mother=mother,
        father=father,
        birth_date=birth_date,
        birth_time=birth_time,
        birth_place=habitation,
        declaration_date=decl_date,
        declaration_time=decl_time,
        declarant_name=decl_name,
        declarant_age=decl_age,
        declarant_occupation=decl_occ,
        owner_name=owner,
        habitation_name=habitation,
        officer_name=officer,
        commune=commune,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RegexExtractor:
    """Deterministic regex-based NER extractor for civil registry acts."""

    @staticmethod
    def detect_act_type_from_marge(marge_text: str) -> Optional[str]:
        """Detect act type from Marge text using keywords.

        Returns 'deces', 'naissance', 'mariage', or None if not detected.
        """
        if not marge_text or not marge_text.strip():
            return None
        text = marge_text.lower()
        if re.search(r"d[ée]c[eè]s", text):
            return "deces"
        if re.search(r"naissance|accouch[ée]|\bné(e|é)", text):
            return "naissance"
        if re.search(r"mariage", text):
            return "mariage"
        return None

    @staticmethod
    def detect_act_type_from_plein_texte(plein_texte_text: str) -> Optional[str]:
        """Detect act type from Plein Texte using formulaic indicators.

        Returns 'deces', 'naissance', or None if not detected.
        Death: "y est décédé(e)"
        Birth: "est accouchée"
        """
        if not plein_texte_text:
            return None
        has_death = _DEATH_INDICATOR.search(plein_texte_text)
        has_birth = _BIRTH_INDICATOR.search(plein_texte_text)
        if has_death and not has_birth:
            return "deces"
        if has_birth and not has_death:
            return "naissance"
        # Both or neither — ambiguous
        if has_death and has_birth:
            # Birth indicator is more specific (accouchée), prefer it
            return "naissance"
        return None

    def detect_act_type(self, record: ActRecord) -> str:
        """Detect act type using both Marge and Plein Texte.

        Priority: Marge (explicit label) > Plein Texte (body indicators).
        Returns 'deces', 'naissance', 'mariage', or 'unknown'.
        """
        marge_type = self.detect_act_type_from_marge(record.marge_text)
        if marge_type:
            return marge_type
        plein_texte_type = self.detect_act_type_from_plein_texte(
            record.plein_texte_text
        )
        if plein_texte_type:
            return plein_texte_type
        return "unknown"

    @staticmethod
    def extract_act_number_from_marge(marge_text: str) -> Optional[str]:
        """Extract act number from Marge (e.g. 'N° 78' -> '78')."""
        if not marge_text:
            return None
        match = re.search(r"[Nn][°ᵒo˚.\s\^]*\s*(\d+)", marge_text)
        return match.group(1) if match else None

    @staticmethod
    def extract_name_from_marge(marge_text: str) -> Optional[str]:
        """Extract person name from Marge.

        Patterns:
          - "Décès de Françoise"
          - "Naissance de Joséphine"
          - "Camillette appt au Sr Robine"  (name at start)
          - "Volcidor, fils de claire"  (name at start)
        """
        if not marge_text:
            return None
        # Pattern 1: "décès/naissance [N° X\n] de [NAME]" or "d'[NAME]"
        # Allow act number and newlines between keyword and "de"
        match = re.search(
            r"(?:d[ée]c[eè]s|naissance)(?:\s+[Nn][°ᵒo˚.\s]*\s*\d+)?\s+(?:de\s+|d['\u2019])(.+?)(?:\s*,|\s+au\s+|\s+appt|\s+apt|\s+appartenant|\s+h[.\^]|\s+bon\s|\s+ag[ée]+\s|\s+enf[.ᵗ\^ant]|\s+fils\s|\s+fille\s|$)",
            marge_text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            name = re.sub(r"\s+", " ", match.group(1).strip()).rstrip(",.")
            # Filter generic descriptions (not actual names)
            if re.match(r"^(?:un[e]?\s+|une\s+|la\s+|le\s+)", name, re.IGNORECASE):
                return None
            if len(name) < 2:
                return None
            return name

        # Pattern 2: name at start of Marge (before "appt" or "décédé" or "née")
        lines = marge_text.strip().splitlines()
        first_line = lines[0].strip()
        # Skip if first line is just "N° XX" or "Décès" / "Naissance"
        if re.match(r"^[Nn][°ᵒo˚.\s]*\s*\d+", first_line):
            return None
        if re.match(
            r"^(?:d[ée]c[eè]s|naissance|mariage)\s*$", first_line, re.IGNORECASE
        ):
            return None

        # Check if first line has a name followed by qualifier
        match = re.match(
            r"^(.+?)(?:\s*,?\s*(?:appt|apt|appartenant|fils|fille|d[ée]c[ée]d|n[ée]e?\s+le))",
            first_line,
            re.IGNORECASE,
        )
        if match:
            name = re.sub(r"\s+", " ", match.group(1).strip()).rstrip(",.")
            if len(name) >= 2:
                return name

        return None

    @staticmethod
    def extract_owner_from_marge(marge_text: str) -> Optional[str]:
        """Extract owner name from Marge.

        Patterns:
          - "au Sr/Sieur [NAME]"
          - "appt/appartenant au Sr [NAME]"
          - "Héritier/Hrs [NAME]"
          - "à Mr/Mme [NAME]"
        """
        if not marge_text:
            return None
        # Pattern: "au Sr/Sieur/Sᵗ/Sʳ/S^r/St [NAME]" or "à Mr/Mme [NAME]" or "Ve/Vve [NAME]"
        match = re.search(
            r"(?:au\s+(?:Sr\.?|St\.?|Sieur|S[ᵗʳ^.\s][\w]*)|[àa]\s+(?:Mr\.?|Mme\.?|Monsieur|Madame|Dame|Demoiselle|la\s+(?:Dlle|Dame|V[eᵉ]|Vve)\.?)|V[eᵉ]\s|Vve\s)\s*(.+?)(?:\s*[,;.]|\s*$)",
            marge_text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            owner = re.sub(r"\s+", " ", match.group(1).strip()).rstrip(",.")
            if len(owner) >= 2:
                return owner

        # Pattern: "Héritier(s)/Hrs [NAME]"
        match = re.search(
            r"(?:h[ée]ritiers?|hrs?\.?,?)\s+(.+?)(?:\s*[,;.]|\s*$)",
            marge_text,
            re.IGNORECASE,
        )
        if match:
            owner = re.sub(r"\s+", " ", match.group(1).strip()).rstrip(",.")
            if len(owner) >= 2:
                return owner

        # Pattern: "bon/h^on/habitation [NAME]" (habitation = owner reference in Marge)
        match = re.search(
            r"(?:bon|h\^on|h[.\s]?on|habi?t?[.\s])\s+(.+?)(?:\s*[,;.]|\s*$)",
            marge_text,
            re.IGNORECASE,
        )
        if match:
            owner = re.sub(r"\s+", " ", match.group(1).strip()).rstrip(",.")
            if len(owner) >= 2:
                return owner

        return None

    def extract(self, record: ActRecord) -> NERResult:
        """Extract entities from a single act record."""
        # Detect act type from Marge and Plein Texte
        marge_act_type = self.detect_act_type_from_marge(record.marge_text)
        act_type = (
            marge_act_type
            or self.detect_act_type_from_plein_texte(record.plein_texte_text)
            or "unknown"
        )

        # Extract Marge fields
        marge_act_number = self.extract_act_number_from_marge(record.marge_text)
        marge_act_name = self.extract_name_from_marge(record.marge_text)
        marge_act_owner = self.extract_owner_from_marge(record.marge_text)

        death_act = None
        birth_act = None

        if act_type == "deces":
            death_act = _extract_death(record)
        elif act_type == "naissance":
            birth_act = _extract_birth(record)
        # mariage: no extractor yet

        return NERResult(
            act_id=record.act_id,
            act_type=act_type,
            extraction_method="regex",
            marge_act_type=marge_act_type,
            marge_act_name=marge_act_name,
            marge_act_number=marge_act_number,
            marge_act_owner=marge_act_owner,
            death_act=death_act,
            birth_act=birth_act,
            raw_marge=record.marge_text,
            raw_plein_texte=record.plein_texte_text,
        )

    def extract_all(self, records: List[ActRecord]) -> List[NERResult]:
        """Extract entities from all act records."""
        results = []
        for record in records:
            try:
                result = self.extract(record)
                results.append(result)
            except Exception as e:
                logger.error("Failed to extract %s: %s", record.act_id, e)
        return results
