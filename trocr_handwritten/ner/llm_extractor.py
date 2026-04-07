"""LLM-based NER extraction for civil registry acts using Gemini.

Uses function calling to extract structured entities from OCR-transcribed
acts, reusing the existing LLM provider infrastructure.
"""

import asyncio
import json
import logging
from typing import List, Optional

from tqdm.asyncio import tqdm_asyncio

from trocr_handwritten.llm.factory import get_provider
from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.ner.schemas import (
    ActRecord,
    BirthActEntity,
    ChildInfo,
    DeathActEntity,
    MarriageActEntity,
    NERResult,
    PersonInfo,
)
from trocr_handwritten.utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Function-calling schemas (derived from Pydantic models)
# ---------------------------------------------------------------------------


_MARGE_FIELDS = {
    "marge_act_type": {
        "type": ["string", "null"],
        "description": "Type d'acte tel qu'écrit dans la marge (ex: 'Naissance', 'Décès', 'Mariage')",
    },
    "marge_act_name": {
        "type": ["string", "null"],
        "description": "Nom de l'esclave tel qu'écrit dans la marge",
    },
    "marge_act_number": {
        "type": ["string", "null"],
        "description": "Numéro de l'acte tel qu'écrit dans la marge (ex: '1', '44')",
    },
    "marge_act_owner": {
        "type": ["string", "null"],
        "description": "Nom du propriétaire tel qu'écrit dans la marge",
    },
}


def _extract_marge_fields(raw_json: str) -> dict:
    """Extract marge fields from parsed tool JSON."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return {}
    return {
        "marge_act_type": data.get("marge_act_type"),
        "marge_act_name": data.get("marge_act_name"),
        "marge_act_number": data.get("marge_act_number"),
        "marge_act_owner": data.get("marge_act_owner"),
    }


def _build_death_tool() -> dict:
    """Build OpenAI function-calling tool definition for death act extraction."""
    return {
        "type": "function",
        "function": {
            "name": "extract_death_act",
            "description": "Extract entities from a death act (acte de décès)",
            "parameters": {
                "type": "object",
                "properties": {
                    **_MARGE_FIELDS,
                    "person_name": {
                        "type": ["string", "null"],
                        "description": "Nom complet de la personne esclave décédée",
                    },
                    "person_sex": {
                        "type": ["string", "null"],
                        "enum": ["homme", "femme", None],
                    },
                    "person_qualifier": {
                        "type": ["string", "null"],
                        "description": "Qualificatif racial de l'esclave tel qu'écrit : nègre, négresse, mulâtre, quarteron, rouge, etc.",
                    },
                    "person_age": {
                        "type": ["string", "null"],
                        "description": "Âge (en chiffres)",
                    },
                    "person_occupation": {"type": ["string", "null"]},
                    "person_registration_register": {
                        "type": ["string", "null"],
                        "description": "Lettre du registre (A, B, C, D)",
                    },
                    "person_registration_number": {
                        "type": ["string", "null"],
                        "description": "Numéro d'immatriculation",
                    },
                    "death_date": {
                        "type": ["string", "null"],
                        "description": "Date du décès (format original du document)",
                    },
                    "death_place": {
                        "type": ["string", "null"],
                        "description": "Lieu du décès (nom de l'habitation)",
                    },
                    "declaration_date": {
                        "type": ["string", "null"],
                        "description": "Date de la déclaration (format original du document)",
                    },
                    "declarant_name": {
                        "type": ["string", "null"],
                        "description": "Nom complet du déclarant",
                    },
                    "declarant_age": {
                        "type": ["string", "null"],
                        "description": "Âge du déclarant (en chiffres)",
                    },
                    "declarant_occupation": {
                        "type": ["string", "null"],
                        "description": "Profession du déclarant",
                    },
                    "owner_name": {
                        "type": ["string", "null"],
                        "description": "Nom du propriétaire de l'esclave",
                    },
                    "owner_commune": {
                        "type": ["string", "null"],
                        "description": "Nom de la commune du propriétaire",
                    },
                    "owner_residence": {
                        "type": ["string", "null"],
                        "description": "Nom de la résidence du propriétaire (e.g nom de plantation, rue, etc.)",
                    },
                    "habitation_name": {
                        "type": ["string", "null"],
                        "description": "Nom de l'habitation/plantation ou résidait l'esclave",
                    },
                    "officer_name": {
                        "type": ["string", "null"],
                        "description": "Nom de l'officier de l'état civil (maire ou adjoint)",
                    },
                    "commune": {
                        "type": ["string", "null"],
                        "description": "Nom de la commune",
                    },
                },
                "required": [
                    "person_name",
                    "person_sex",
                    "person_age",
                    "person_qualifier",
                    "person_registration_register",
                    "person_registration_number",
                    "habitation_name",
                    "owner_name",
                    "declarant_name",
                ],
            },
        },
    }


def _build_birth_tool() -> dict:
    """Build OpenAI function-calling tool definition for birth act extraction."""
    return {
        "type": "function",
        "function": {
            "name": "extract_birth_act",
            "description": "Extract entities from a birth act (acte de naissance) of a slave child",
            "parameters": {
                "type": "object",
                "properties": {
                    **_MARGE_FIELDS,
                    "child_name": {
                        "type": ["string", "null"],
                        "description": "Nom de l'enfant",
                    },
                    "child_sex": {
                        "type": ["string", "null"],
                        "enum": ["homme", "femme", None],
                    },
                    "child_qualifier": {
                        "type": ["string", "null"],
                        "description": "Qualificatif racial tel qu'écrit : nègre, négresse, mulâtre, quarteron, rouge, etc.",
                    },
                    "child_registration_register": {"type": ["string", "null"]},
                    "child_registration_number": {"type": ["string", "null"]},
                    "mother_name": {
                        "type": ["string", "null"],
                        "description": "Nom de la mère",
                    },
                    "mother_age": {"type": ["string", "null"]},
                    "mother_qualifier": {
                        "type": ["string", "null"],
                        "description": "Qualificatif racial tel qu'écrit : négresse, mulâtresse, quarteronne, rouge, etc.",
                    },
                    "mother_occupation": {"type": ["string", "null"]},
                    "mother_registration_register": {"type": ["string", "null"]},
                    "mother_registration_number": {"type": ["string", "null"]},
                    "father_name": {
                        "type": ["string", "null"],
                        "description": "Nom du père (souvent absent)",
                    },
                    "father_age": {"type": ["string", "null"]},
                    "birth_date": {"type": ["string", "null"]},
                    "birth_place": {
                        "type": ["string", "null"],
                        "description": "Lieu de naissance (habitation)",
                    },
                    "declaration_date": {"type": ["string", "null"]},
                    "declarant_name": {"type": ["string", "null"]},
                    "declarant_age": {"type": ["string", "null"]},
                    "declarant_occupation": {"type": ["string", "null"]},
                    "owner_name": {"type": ["string", "null"]},
                    "habitation_name": {"type": ["string", "null"]},
                    "owner_commune": {"type": ["string", "null"]},
                    "owner_residence": {
                        "type": ["string", "null"],
                        "description": "Nom de la résidence du propriétaire (e.g nom de plantation, rue, etc.)",
                    },
                    "officer_name": {
                        "type": ["string", "null"],
                        "description": "Nom de l'officier de l'état civil (maire ou adjoint)",
                    },
                    "commune": {"type": ["string", "null"]},
                },
                "required": [
                    "child_name",
                    "child_sex",
                    "child_qualifier",
                    "child_registration_register",
                    "child_registration_number",
                    "mother_name",
                    "mother_age",
                    "mother_qualifier",
                    "habitation_name",
                    "owner_name",
                    "declarant_name",
                ],
            },
        },
    }


def _build_marriage_tool() -> dict:
    """Build OpenAI function-calling tool definition for marriage act extraction."""
    return {
        "type": "function",
        "function": {
            "name": "extract_marriage_act",
            "description": "Extract entities from a marriage act (acte de mariage)",
            "parameters": {
                "type": "object",
                "properties": {
                    **_MARGE_FIELDS,
                    "spouse1_name": {
                        "type": ["string", "null"],
                        "description": "Nom complet du premier esclave (généralement l'homme)",
                    },
                    "spouse1_age": {
                        "type": ["string", "null"],
                        "description": "Âge du premier esclave (en chiffres)",
                    },
                    "spouse1_qualifier": {
                        "type": ["string", "null"],
                        "description": "Qualificatif racial tel qu'écrit : nègre, mulâtre, rouge, etc.",
                    },
                    "spouse1_occupation": {
                        "type": ["string", "null"],
                        "description": "Profession du premier esclave (ex: cultivateur)",
                    },
                    "spouse1_registration_register": {
                        "type": ["string", "null"],
                        "description": "Lettre du registre d'immatriculation du premier esclave (A, B, C, D...)",
                    },
                    "spouse1_registration_number": {
                        "type": ["string", "null"],
                        "description": "Numéro d'immatriculation du premier esclave",
                    },
                    "spouse2_name": {
                        "type": ["string", "null"],
                        "description": "Nom complet du second esclave (généralement la femme). OBLIGATOIRE — toujours présent dans le texte.",
                    },
                    "spouse2_age": {
                        "type": ["string", "null"],
                        "description": "Âge du second esclave (en chiffres)",
                    },
                    "spouse2_qualifier": {
                        "type": ["string", "null"],
                        "description": "Qualificatif racial tel qu'écrit : négresse, mulâtresse, rouge, etc.",
                    },
                    "spouse2_occupation": {
                        "type": ["string", "null"],
                        "description": "Profession du second esclave (ex: cultivatrice)",
                    },
                    "spouse2_registration_register": {
                        "type": ["string", "null"],
                        "description": "Lettre du registre d'immatriculation du second esclave (A, B, C, D...)",
                    },
                    "spouse2_registration_number": {
                        "type": ["string", "null"],
                        "description": "Numéro d'immatriculation du second esclave",
                    },
                    "marriage_date": {"type": ["string", "null"]},
                    "declaration_date": {"type": ["string", "null"]},
                    "declarant_name": {"type": ["string", "null"]},
                    "declarant_age": {"type": ["string", "null"]},
                    "declarant_occupation": {"type": ["string", "null"]},
                    "owner_name": {"type": ["string", "null"]},
                    "owner_commune": {"type": ["string", "null"]},
                    "owner_residence": {"type": ["string", "null"]},
                    "habitation_name": {"type": ["string", "null"]},
                    "officer_name": {"type": ["string", "null"]},
                    "commune": {"type": ["string", "null"]},
                },
                "required": [
                    "spouse1_name",
                    "spouse1_age",
                    "spouse1_registration_number",
                    "spouse2_name",
                    "spouse2_age",
                    "spouse2_registration_number",
                    "owner_name",
                    "declarant_name",
                ],
            },
        },
    }


def _build_unknown_tool() -> dict:
    """Build OpenAI function-calling tool definition for unknown act type (union of all fields)."""
    return {
        "type": "function",
        "function": {
            "name": "extract_unknown_act",
            "description": "Identify act type and extract entities from an act of unknown type",
            "parameters": {
                "type": "object",
                "properties": {
                    **_MARGE_FIELDS,
                    "act_type": {
                        "type": "string",
                        "enum": ["deces", "naissance", "mariage"],
                        "description": "Type d'acte détecté",
                    },
                    "person_name": {"type": ["string", "null"]},
                    "person_sex": {
                        "type": ["string", "null"],
                        "enum": ["homme", "femme", None],
                    },
                    "person_qualifier": {"type": ["string", "null"]},
                    "person_age": {"type": ["string", "null"]},
                    "person_occupation": {"type": ["string", "null"]},
                    "person_registration_register": {"type": ["string", "null"]},
                    "person_registration_number": {"type": ["string", "null"]},
                    "event_date": {
                        "type": ["string", "null"],
                        "description": "Date de l'événement (décès, naissance, ou mariage)",
                    },
                    "event_place": {"type": ["string", "null"]},
                    "declaration_date": {"type": ["string", "null"]},
                    "declarant_name": {"type": ["string", "null"]},
                    "declarant_age": {"type": ["string", "null"]},
                    "declarant_occupation": {"type": ["string", "null"]},
                    "owner_name": {"type": ["string", "null"]},
                    "owner_commune": {"type": ["string", "null"]},
                    "owner_residence": {"type": ["string", "null"]},
                    "habitation_name": {"type": ["string", "null"]},
                    "officer_name": {"type": ["string", "null"]},
                    "commune": {"type": ["string", "null"]},
                    "mother_name": {"type": ["string", "null"]},
                    "mother_age": {"type": ["string", "null"]},
                    "mother_qualifier": {"type": ["string", "null"]},
                    "mother_registration_register": {"type": ["string", "null"]},
                    "mother_registration_number": {"type": ["string", "null"]},
                },
                "required": ["act_type", "person_name", "owner_name", "declarant_name"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Parse LLM response into Pydantic models
# ---------------------------------------------------------------------------


def _parse_death_response(raw_json: str) -> Optional[DeathActEntity]:
    """Parse function-calling JSON into a DeathActEntity."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.error("Failed to parse death act JSON: %s", raw_json[:200])
        return None

    person = PersonInfo(
        name=data.get("person_name"),
        sex=data.get("person_sex"),
        qualifier=data.get("person_qualifier"),
        age=data.get("person_age"),
        occupation=data.get("person_occupation"),
        registration_register=data.get("person_registration_register"),
        registration_number=data.get("person_registration_number"),
    )
    return DeathActEntity(
        person=person,
        death_date=data.get("death_date"),
        death_place=data.get("death_place"),
        declaration_date=data.get("declaration_date"),
        declarant_name=data.get("declarant_name"),
        declarant_age=data.get("declarant_age"),
        declarant_occupation=data.get("declarant_occupation"),
        owner_name=data.get("owner_name"),
        owner_commune=data.get("owner_commune"),
        owner_residence=data.get("owner_residence"),
        habitation_name=data.get("habitation_name"),
        officer_name=data.get("officer_name"),
        commune=data.get("commune"),
    )


def _parse_marriage_response(raw_json: str) -> Optional[MarriageActEntity]:
    """Parse function-calling JSON into a MarriageActEntity."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.error("Failed to parse marriage act JSON: %s", raw_json[:200])
        return None

    spouse1 = PersonInfo(
        name=data.get("spouse1_name"),
        sex="homme",
        qualifier=data.get("spouse1_qualifier"),
        age=data.get("spouse1_age"),
        occupation=data.get("spouse1_occupation"),
        registration_register=data.get("spouse1_registration_register"),
        registration_number=data.get("spouse1_registration_number"),
    )
    spouse2 = PersonInfo(
        name=data.get("spouse2_name"),
        sex="femme",
        qualifier=data.get("spouse2_qualifier"),
        age=data.get("spouse2_age"),
        occupation=data.get("spouse2_occupation"),
        registration_register=data.get("spouse2_registration_register"),
        registration_number=data.get("spouse2_registration_number"),
    )
    return MarriageActEntity(
        spouse1=spouse1,
        spouse2=spouse2,
        marriage_date=data.get("marriage_date"),
        declaration_date=data.get("declaration_date"),
        declarant_name=data.get("declarant_name"),
        declarant_age=data.get("declarant_age"),
        declarant_occupation=data.get("declarant_occupation"),
        owner_name=data.get("owner_name"),
        owner_commune=data.get("owner_commune"),
        owner_residence=data.get("owner_residence"),
        habitation_name=data.get("habitation_name"),
        officer_name=data.get("officer_name"),
        commune=data.get("commune"),
    )


def _parse_unknown_response(
    raw_json: str,
) -> tuple[
    Optional[str],
    Optional[DeathActEntity],
    Optional[BirthActEntity],
    Optional[MarriageActEntity],
]:
    """Parse unknown act JSON; returns (detected_act_type, death, birth, marriage)."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.error("Failed to parse unknown act JSON: %s", raw_json[:200])
        return None, None, None, None

    act_type = data.get("act_type")

    if act_type == "deces":
        person = PersonInfo(
            name=data.get("person_name"),
            sex=data.get("person_sex"),
            qualifier=data.get("person_qualifier"),
            age=data.get("person_age"),
            occupation=data.get("person_occupation"),
            registration_register=data.get("person_registration_register"),
            registration_number=data.get("person_registration_number"),
        )
        entity = DeathActEntity(
            person=person,
            death_date=data.get("event_date"),
            death_place=data.get("event_place"),
            declaration_date=data.get("declaration_date"),
            declarant_name=data.get("declarant_name"),
            declarant_age=data.get("declarant_age"),
            declarant_occupation=data.get("declarant_occupation"),
            owner_name=data.get("owner_name"),
            owner_commune=data.get("owner_commune"),
            owner_residence=data.get("owner_residence"),
            habitation_name=data.get("habitation_name"),
            officer_name=data.get("officer_name"),
            commune=data.get("commune"),
        )
        return act_type, entity, None, None

    if act_type == "naissance":
        child = ChildInfo(
            name=data.get("person_name"),
            sex=data.get("person_sex"),
            qualifier=data.get("person_qualifier"),
            registration_register=data.get("person_registration_register"),
            registration_number=data.get("person_registration_number"),
        )
        mother = PersonInfo(
            name=data.get("mother_name"),
            sex="femme" if data.get("mother_name") else None,
            qualifier=data.get("mother_qualifier"),
            age=data.get("mother_age"),
            registration_register=data.get("mother_registration_register"),
            registration_number=data.get("mother_registration_number"),
        )
        entity = BirthActEntity(
            child=child,
            mother=mother,
            birth_date=data.get("event_date"),
            birth_place=data.get("event_place"),
            declaration_date=data.get("declaration_date"),
            declarant_name=data.get("declarant_name"),
            declarant_age=data.get("declarant_age"),
            declarant_occupation=data.get("declarant_occupation"),
            owner_name=data.get("owner_name"),
            owner_commune=data.get("owner_commune"),
            owner_residence=data.get("owner_residence"),
            habitation_name=data.get("habitation_name"),
            officer_name=data.get("officer_name"),
            commune=data.get("commune"),
        )
        return act_type, None, entity, None

    if act_type == "mariage":
        spouse1 = PersonInfo(name=data.get("person_name"), sex="homme")
        spouse2 = PersonInfo(name=data.get("mother_name"), sex="femme")
        entity = MarriageActEntity(
            spouse1=spouse1,
            spouse2=spouse2,
            marriage_date=data.get("event_date"),
            declaration_date=data.get("declaration_date"),
            declarant_name=data.get("declarant_name"),
            declarant_age=data.get("declarant_age"),
            declarant_occupation=data.get("declarant_occupation"),
            owner_name=data.get("owner_name"),
            owner_commune=data.get("owner_commune"),
            owner_residence=data.get("owner_residence"),
            habitation_name=data.get("habitation_name"),
            officer_name=data.get("officer_name"),
            commune=data.get("commune"),
        )
        return act_type, None, None, entity

    return act_type, None, None, None


def _parse_birth_response(raw_json: str) -> Optional[BirthActEntity]:
    """Parse function-calling JSON into a BirthActEntity."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.error("Failed to parse birth act JSON: %s", raw_json[:200])
        return None

    child = ChildInfo(
        name=data.get("child_name"),
        sex=data.get("child_sex"),
        qualifier=data.get("child_qualifier"),
        registration_register=data.get("child_registration_register"),
        registration_number=data.get("child_registration_number"),
    )
    mother = PersonInfo(
        name=data.get("mother_name"),
        sex="femme" if data.get("mother_name") else None,
        qualifier=data.get("mother_qualifier"),
        age=data.get("mother_age"),
        occupation=data.get("mother_occupation"),
        registration_register=data.get("mother_registration_register"),
        registration_number=data.get("mother_registration_number"),
    )
    father = PersonInfo(
        name=data.get("father_name"),
        age=data.get("father_age"),
    )
    return BirthActEntity(
        child=child,
        mother=mother,
        father=father,
        birth_date=data.get("birth_date"),
        birth_place=data.get("birth_place"),
        declaration_date=data.get("declaration_date"),
        declarant_name=data.get("declarant_name"),
        declarant_age=data.get("declarant_age"),
        declarant_occupation=data.get("declarant_occupation"),
        owner_name=data.get("owner_name"),
        owner_commune=data.get("owner_commune"),
        owner_residence=data.get("owner_residence"),
        habitation_name=data.get("habitation_name"),
        officer_name=data.get("officer_name"),
        commune=data.get("commune"),
    )


# ---------------------------------------------------------------------------
# LLM Extractor
# ---------------------------------------------------------------------------


class LLMExtractor:
    """Async LLM-based NER extractor using function calling."""

    def __init__(
        self,
        settings: LLMSettings,
        death_prompt_path: str = "config/ner_death.prompt",
        birth_prompt_path: str = "config/ner_birth.prompt",
        marriage_prompt_path: str = "config/ner_marriage.prompt",
        unknown_prompt_path: str = "config/ner_unknown.prompt",
    ):
        self.provider = get_provider(settings)
        self.cost_tracker = CostTracker(model_name=settings.model_name)
        self.death_prompt = _load_prompt(death_prompt_path)
        self.birth_prompt = _load_prompt(birth_prompt_path)
        self.marriage_prompt = _load_prompt(marriage_prompt_path)
        self.unknown_prompt = _load_prompt(unknown_prompt_path)
        self.death_tool = _build_death_tool()
        self.birth_tool = _build_birth_tool()
        self.marriage_tool = _build_marriage_tool()
        self.unknown_tool = _build_unknown_tool()
        self.failed: dict = {}

    async def extract(self, record: ActRecord) -> NERResult:
        """Extract entities from a single act record using the LLM."""
        # Build input text combining marge + plein texte
        user_text = (
            f"MARGE:\n{record.marge_text}\n\nTEXTE COMPLET:\n{record.plein_texte_text}"
        )

        death_act = None
        birth_act = None
        marriage_act = None
        marge_fields: dict = {}

        if record.act_type == "deces":
            raw_json, inp, out, think = await self.provider.call_text_async(
                user_text,
                self.death_prompt,
                tools=[self.death_tool],
                tool_choice="required",
            )
            self.cost_tracker.add_usage(inp, out, think)
            if raw_json:
                marge_fields = _extract_marge_fields(raw_json)
                death_act = _parse_death_response(raw_json)

        elif record.act_type == "naissance":
            raw_json, inp, out, think = await self.provider.call_text_async(
                user_text,
                self.birth_prompt,
                tools=[self.birth_tool],
                tool_choice="required",
            )
            self.cost_tracker.add_usage(inp, out, think)
            if raw_json:
                marge_fields = _extract_marge_fields(raw_json)
                birth_act = _parse_birth_response(raw_json)

        elif record.act_type == "mariage":
            raw_json, inp, out, think = await self.provider.call_text_async(
                user_text,
                self.marriage_prompt,
                tools=[self.marriage_tool],
                tool_choice="required",
            )
            self.cost_tracker.add_usage(inp, out, think)
            if raw_json:
                marge_fields = _extract_marge_fields(raw_json)
                marriage_act = _parse_marriage_response(raw_json)

        else:
            raw_json, inp, out, think = await self.provider.call_text_async(
                user_text,
                self.unknown_prompt,
                tools=[self.unknown_tool],
                tool_choice="required",
            )
            self.cost_tracker.add_usage(inp, out, think)
            if raw_json:
                marge_fields = _extract_marge_fields(raw_json)
                detected_type, death_act, birth_act, marriage_act = (
                    _parse_unknown_response(raw_json)
                )
                if detected_type:
                    record = record.model_copy(update={"act_type": detected_type})

        return NERResult(
            act_id=record.act_id,
            act_type=record.act_type,
            extraction_method="llm",
            marge_act_type=marge_fields.get("marge_act_type"),
            marge_act_name=marge_fields.get("marge_act_name"),
            marge_act_number=marge_fields.get("marge_act_number"),
            marge_act_owner=marge_fields.get("marge_act_owner"),
            death_act=death_act,
            birth_act=birth_act,
            marriage_act=marriage_act,
            raw_marge=record.marge_text,
            raw_plein_texte=record.plein_texte_text,
        )

    async def extract_batch(
        self,
        records: List[ActRecord],
        max_concurrent: int = 10,
    ) -> List[NERResult]:
        """Extract entities from all records with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _process(record: ActRecord) -> NERResult:
            async with semaphore:
                try:
                    return await self.extract(record)
                except Exception as e:
                    logger.error("Failed to extract %s: %s", record.act_id, e)
                    self.failed[record.act_id] = str(e)
                    return NERResult(
                        act_id=record.act_id,
                        act_type=record.act_type,
                        extraction_method="llm",
                        raw_marge=record.marge_text,
                        raw_plein_texte=record.plein_texte_text,
                    )

        tasks = [_process(r) for r in records]
        results = await tqdm_asyncio.gather(*tasks, desc="LLM NER extraction")
        return list(results)
