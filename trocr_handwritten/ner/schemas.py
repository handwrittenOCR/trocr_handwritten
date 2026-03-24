"""Pydantic models for NER extraction from civil registry acts.

Shared by both regex and LLM extractors. Adapted to slave civil registries
from Guadeloupe (1841-1848): first-name-only identification, registration
numbers, owner/habitation references, "parents inconnus".
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal, List

# ---------------------------------------------------------------------------
# Input: one reconstructed act from paired Marge + Plein Texte crops
# ---------------------------------------------------------------------------


class ActRecord(BaseModel):
    """A single civil act reconstructed from YOLO crops."""

    act_id: str = Field(
        description="Unique identifier, e.g. abymes_1842_page005_order1"
    )
    act_type: Literal["deces", "naissance", "mariage", "unknown"] = Field(
        description="Act type detected from Marge text"
    )
    act_number: Optional[str] = Field(
        default=None, description="Act number from Marge, e.g. '2', '44'"
    )
    marge_text: str = Field(description="Full text from the Marge crop")
    plein_texte_text: str = Field(description="Full text from the Plein Texte crop")
    source_page: str = Field(description="Page folder name")
    source_marge_file: Optional[str] = Field(
        default=None, description="Marge .md filename"
    )
    source_plein_texte_file: str = Field(description="Plein Texte .md filename")
    commune: str = Field(description="Commune name, e.g. 'abymes'")
    year: str = Field(description="Year, e.g. '1842'")
    order_on_page: int = Field(description="Reading order position on the page")


# ---------------------------------------------------------------------------
# Extracted entities
# ---------------------------------------------------------------------------


class PersonInfo(BaseModel):
    """Information about a person mentioned in the act."""

    name: Optional[str] = Field(default=None, description="Full name as written")
    sex: Optional[Literal["homme", "femme"]] = Field(default=None)
    age: Optional[str] = Field(
        default=None, description="Age as written or converted to number"
    )
    occupation: Optional[str] = Field(default=None, description="Profession/job")
    registration_register: Optional[str] = Field(
        default=None, description="Registration register letter, e.g. 'D', 'C'"
    )
    registration_number: Optional[str] = Field(
        default=None, description="Registration number, e.g. '3328'"
    )


class DeathActEntity(BaseModel):
    """Extracted entities from a death act (acte de deces)."""

    person: PersonInfo = Field(default_factory=PersonInfo)
    death_date: Optional[str] = Field(default=None, description="Date of death")
    death_time: Optional[str] = Field(default=None, description="Time of death")
    death_place: Optional[str] = Field(
        default=None, description="Place of death (habitation name)"
    )
    declaration_date: Optional[str] = Field(default=None)
    declaration_time: Optional[str] = Field(default=None)
    declarant_name: Optional[str] = Field(default=None)
    declarant_age: Optional[str] = Field(default=None)
    declarant_occupation: Optional[str] = Field(default=None)
    owner_name: Optional[str] = Field(
        default=None, description="Owner of the enslaved person"
    )
    habitation_name: Optional[str] = Field(default=None)
    commune: Optional[str] = Field(default=None)


class BirthActEntity(BaseModel):
    """Extracted entities from a birth act (acte de naissance)."""

    child: PersonInfo = Field(default_factory=PersonInfo)
    mother: PersonInfo = Field(default_factory=PersonInfo)
    father: PersonInfo = Field(default_factory=PersonInfo)
    birth_date: Optional[str] = Field(default=None, description="Date of birth")
    birth_time: Optional[str] = Field(default=None, description="Time of birth")
    birth_place: Optional[str] = Field(
        default=None, description="Place of birth (habitation name)"
    )
    declaration_date: Optional[str] = Field(default=None)
    declaration_time: Optional[str] = Field(default=None)
    declarant_name: Optional[str] = Field(default=None)
    declarant_age: Optional[str] = Field(default=None)
    declarant_occupation: Optional[str] = Field(default=None)
    owner_name: Optional[str] = Field(default=None)
    habitation_name: Optional[str] = Field(default=None)
    commune: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Output: unified result from either extractor
# ---------------------------------------------------------------------------


class NERResult(BaseModel):
    """Result of NER extraction for a single act."""

    act_id: str
    act_type: Literal["deces", "naissance", "mariage", "unknown"]
    extraction_method: Literal["regex", "llm"]
    death_act: Optional[DeathActEntity] = None
    birth_act: Optional[BirthActEntity] = None
    raw_marge: str = Field(description="Original Marge text")
    raw_plein_texte: str = Field(description="Original Plein Texte text")


# ---------------------------------------------------------------------------
# Helpers for flattening to CSV
# ---------------------------------------------------------------------------

DEATH_CSV_COLUMNS: List[str] = [
    "act_id",
    "act_type",
    "extraction_method",
    "person_name",
    "person_sex",
    "person_age",
    "person_occupation",
    "person_registration_register",
    "person_registration_number",
    "death_date",
    "death_time",
    "death_place",
    "declaration_date",
    "declaration_time",
    "declarant_name",
    "declarant_age",
    "declarant_occupation",
    "owner_name",
    "habitation_name",
    "commune",
]

BIRTH_CSV_COLUMNS: List[str] = [
    "act_id",
    "act_type",
    "extraction_method",
    "child_name",
    "child_sex",
    "child_age",
    "child_registration_register",
    "child_registration_number",
    "mother_name",
    "mother_sex",
    "mother_age",
    "mother_occupation",
    "mother_registration_register",
    "mother_registration_number",
    "father_name",
    "father_sex",
    "father_age",
    "father_occupation",
    "father_registration_register",
    "father_registration_number",
    "birth_date",
    "birth_time",
    "birth_place",
    "declaration_date",
    "declaration_time",
    "declarant_name",
    "declarant_age",
    "declarant_occupation",
    "owner_name",
    "habitation_name",
    "commune",
]


def flatten_ner_result(result: NERResult) -> dict:
    """Flatten a NERResult into a flat dict suitable for CSV export."""
    row = {
        "act_id": result.act_id,
        "act_type": result.act_type,
        "extraction_method": result.extraction_method,
        "raw_marge": result.raw_marge,
        "raw_plein_texte": result.raw_plein_texte,
    }

    if result.death_act:
        d = result.death_act
        row.update(
            {
                "person_name": d.person.name,
                "person_sex": d.person.sex,
                "person_age": d.person.age,
                "person_occupation": d.person.occupation,
                "person_registration_register": d.person.registration_register,
                "person_registration_number": d.person.registration_number,
                "death_date": d.death_date,
                "death_time": d.death_time,
                "death_place": d.death_place,
                "declaration_date": d.declaration_date,
                "declaration_time": d.declaration_time,
                "declarant_name": d.declarant_name,
                "declarant_age": d.declarant_age,
                "declarant_occupation": d.declarant_occupation,
                "owner_name": d.owner_name,
                "habitation_name": d.habitation_name,
                "commune": d.commune,
            }
        )

    if result.birth_act:
        b = result.birth_act
        row.update(
            {
                "child_name": b.child.name,
                "child_sex": b.child.sex,
                "child_age": b.child.age,
                "child_registration_register": b.child.registration_register,
                "child_registration_number": b.child.registration_number,
                "mother_name": b.mother.name,
                "mother_sex": b.mother.sex,
                "mother_age": b.mother.age,
                "mother_occupation": b.mother.occupation,
                "mother_registration_register": b.mother.registration_register,
                "mother_registration_number": b.mother.registration_number,
                "father_name": b.father.name,
                "father_sex": b.father.sex,
                "father_age": b.father.age,
                "father_occupation": b.father.occupation,
                "father_registration_register": b.father.registration_register,
                "father_registration_number": b.father.registration_number,
                "birth_date": b.birth_date,
                "birth_time": b.birth_time,
                "birth_place": b.birth_place,
                "declaration_date": b.declaration_date,
                "declaration_time": b.declaration_time,
                "declarant_name": b.declarant_name,
                "declarant_age": b.declarant_age,
                "declarant_occupation": b.declarant_occupation,
                "owner_name": b.owner_name,
                "habitation_name": b.habitation_name,
                "commune": b.commune,
            }
        )

    return row
