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
    DeathActEntity,
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
                    "person_name": {
                        "type": ["string", "null"],
                        "description": "Nom complet de la personne décédée",
                    },
                    "person_sex": {
                        "type": ["string", "null"],
                        "enum": ["homme", "femme", None],
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
                        "description": "Date du décès",
                    },
                    "death_time": {
                        "type": ["string", "null"],
                        "description": "Heure du décès",
                    },
                    "death_place": {
                        "type": ["string", "null"],
                        "description": "Lieu du décès (nom de l'habitation)",
                    },
                    "declaration_date": {
                        "type": ["string", "null"],
                        "description": "Date de la déclaration",
                    },
                    "declaration_time": {
                        "type": ["string", "null"],
                        "description": "Heure de la déclaration",
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
                    "habitation_name": {
                        "type": ["string", "null"],
                        "description": "Nom de l'habitation/plantation",
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
            "description": "Extract entities from a birth act (acte de naissance)",
            "parameters": {
                "type": "object",
                "properties": {
                    "child_name": {
                        "type": ["string", "null"],
                        "description": "Nom de l'enfant",
                    },
                    "child_sex": {
                        "type": ["string", "null"],
                        "enum": ["homme", "femme", None],
                    },
                    "child_registration_register": {"type": ["string", "null"]},
                    "child_registration_number": {"type": ["string", "null"]},
                    "mother_name": {
                        "type": ["string", "null"],
                        "description": "Nom de la mère",
                    },
                    "mother_age": {"type": ["string", "null"]},
                    "mother_occupation": {"type": ["string", "null"]},
                    "mother_registration_register": {"type": ["string", "null"]},
                    "mother_registration_number": {"type": ["string", "null"]},
                    "father_name": {
                        "type": ["string", "null"],
                        "description": "Nom du père (souvent absent)",
                    },
                    "father_age": {"type": ["string", "null"]},
                    "birth_date": {"type": ["string", "null"]},
                    "birth_time": {"type": ["string", "null"]},
                    "birth_place": {
                        "type": ["string", "null"],
                        "description": "Lieu de naissance (habitation)",
                    },
                    "declaration_date": {"type": ["string", "null"]},
                    "declaration_time": {"type": ["string", "null"]},
                    "declarant_name": {"type": ["string", "null"]},
                    "declarant_age": {"type": ["string", "null"]},
                    "declarant_occupation": {"type": ["string", "null"]},
                    "owner_name": {"type": ["string", "null"]},
                    "habitation_name": {"type": ["string", "null"]},
                    "officer_name": {
                        "type": ["string", "null"],
                        "description": "Nom de l'officier de l'état civil (maire ou adjoint)",
                    },
                    "commune": {"type": ["string", "null"]},
                },
                "required": [
                    "child_name",
                    "child_sex",
                    "child_registration_register",
                    "child_registration_number",
                    "mother_name",
                    "mother_age",
                    "habitation_name",
                    "owner_name",
                    "declarant_name",
                ],
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
        age=data.get("person_age"),
        occupation=data.get("person_occupation"),
        registration_register=data.get("person_registration_register"),
        registration_number=data.get("person_registration_number"),
    )
    return DeathActEntity(
        person=person,
        death_date=data.get("death_date"),
        death_time=data.get("death_time"),
        death_place=data.get("death_place"),
        declaration_date=data.get("declaration_date"),
        declaration_time=data.get("declaration_time"),
        declarant_name=data.get("declarant_name"),
        declarant_age=data.get("declarant_age"),
        declarant_occupation=data.get("declarant_occupation"),
        owner_name=data.get("owner_name"),
        habitation_name=data.get("habitation_name"),
        officer_name=data.get("officer_name"),
        commune=data.get("commune"),
    )


def _parse_birth_response(raw_json: str) -> Optional[BirthActEntity]:
    """Parse function-calling JSON into a BirthActEntity."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.error("Failed to parse birth act JSON: %s", raw_json[:200])
        return None

    child = PersonInfo(
        name=data.get("child_name"),
        sex=data.get("child_sex"),
        registration_register=data.get("child_registration_register"),
        registration_number=data.get("child_registration_number"),
    )
    mother = PersonInfo(
        name=data.get("mother_name"),
        sex="femme" if data.get("mother_name") else None,
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
        birth_time=data.get("birth_time"),
        birth_place=data.get("birth_place"),
        declaration_date=data.get("declaration_date"),
        declaration_time=data.get("declaration_time"),
        declarant_name=data.get("declarant_name"),
        declarant_age=data.get("declarant_age"),
        declarant_occupation=data.get("declarant_occupation"),
        owner_name=data.get("owner_name"),
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
    ):
        self.provider = get_provider(settings)
        self.cost_tracker = CostTracker(model_name=settings.model_name)
        self.death_prompt = _load_prompt(death_prompt_path)
        self.birth_prompt = _load_prompt(birth_prompt_path)
        self.death_tool = _build_death_tool()
        self.birth_tool = _build_birth_tool()
        self.failed: dict = {}

    async def extract(self, record: ActRecord) -> NERResult:
        """Extract entities from a single act record using the LLM."""
        # Build input text combining marge + plein texte
        user_text = (
            f"MARGE:\n{record.marge_text}\n\nTEXTE COMPLET:\n{record.plein_texte_text}"
        )

        death_act = None
        birth_act = None

        if record.act_type == "deces":
            raw_json, inp, out, _think = await self.provider.call_text_async(
                user_text,
                self.death_prompt,
                tools=[self.death_tool],
                tool_choice="required",
            )
            self.cost_tracker.add_usage(inp, out)
            if raw_json:
                death_act = _parse_death_response(raw_json)

        elif record.act_type == "naissance":
            raw_json, inp, out, _think = await self.provider.call_text_async(
                user_text,
                self.birth_prompt,
                tools=[self.birth_tool],
                tool_choice="required",
            )
            self.cost_tracker.add_usage(inp, out)
            if raw_json:
                birth_act = _parse_birth_response(raw_json)

        else:
            # Unknown type — try death first (most common)
            raw_json, inp, out, _think = await self.provider.call_text_async(
                user_text,
                self.death_prompt,
                tools=[self.death_tool],
                tool_choice="required",
            )
            self.cost_tracker.add_usage(inp, out)
            if raw_json:
                death_act = _parse_death_response(raw_json)

        return NERResult(
            act_id=record.act_id,
            act_type=record.act_type,
            extraction_method="llm",
            death_act=death_act,
            birth_act=birth_act,
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
