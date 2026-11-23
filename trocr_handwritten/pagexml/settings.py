from dataclasses import dataclass, field

COLS = [
    "page",
    "page_xml",
    "split",
    "region_id",
    "class",
    "handwriting",
    "transcription",
    "htr_category",
    "htr_crop",
    "ls_key",
    "x",
    "y",
    "w",
    "h",
    "box_provenance",
]


@dataclass
class PageXmlSettings:
    """
    Default paths of the corpus PAGE-XML, the detection splits, the HTR dataset and the manifest.
    """

    page_xml_dir: str = "data/processed/page_xml"
    splits_dir: str = "data/layout_merged"
    htr_dir: str = "data/ocr/20260807-ft-data"
    manifest: str = "data/processed/pages_manifest.csv"
    splits: list = field(default_factory=lambda: ["train", "dev", "test"])
