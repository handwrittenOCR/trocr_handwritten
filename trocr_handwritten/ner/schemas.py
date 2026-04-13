"""Input schema for NER extraction from paired OCR crops.

Domain-specific entity schemas (fields, act types) belong in the caller project.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ActRecord(BaseModel):
    """A single act reconstructed from paired Marge + Plein Texte OCR crops."""

    act_id: str = Field(description="Unique identifier for this act")
    act_type: Optional[str] = Field(
        default=None, description="Act type label (caller-defined, e.g. 'birth')"
    )
    marge_text: str = Field(description="Full text from the Marge crop")
    plein_texte_text: str = Field(description="Full text from the Plein Texte crop")
    source_page: str = Field(description="Page folder name")
    source_marge_file: Optional[str] = Field(default=None)
    source_plein_texte_file: str = Field(description="Plein Texte .md filename")
    order_on_page: int = Field(description="Reading order position on the page")
