from dataclasses import dataclass, field

CLASSES = ["Plein Texte", "Marge", "Nom", "Section"]


@dataclass
class HwFilterSettings:
    """
    Settings of the handwriting filter.

    The filter classifies a region crop of a transcribed class (Plein Texte, Marge,
    Nom, Section) as handwritten or not, from a frozen image embedding and the
    region class, with a logistic regression.
    """

    manifest: str = "data/processed/pages_manifest.csv"
    images_dir: str = "data/raw/initial"
    out_dir: str = "data/processed/hw_filter"
    embed_model: str = "facebook/dinov2-small"
    image_size: int = 448
    batch_size: int = 32
    C: float = 0.1
    threshold: float = 0.5
    classes: list = field(default_factory=lambda: list(CLASSES))
