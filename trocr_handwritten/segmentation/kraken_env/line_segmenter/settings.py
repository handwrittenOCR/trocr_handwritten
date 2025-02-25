from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path


@dataclass
class MarginSettings:
    """Settings for margin text segmentation"""

    padding: int = 50  # Width padding for margin text
    iou_threshold: float = 1.0  # No IoU filtering for margins
    y_threshold: int = 10  # Vertical threshold for merging lines


@dataclass
class MainTextSettings:
    """Settings for main text segmentation"""

    padding: int = 15  # Height padding for main text
    iou_threshold: float = 0.5  # IoU threshold for filtering overlapping lines
    y_threshold: int = 10  # Vertical threshold for merging lines


@dataclass
class SegmentationSettings:
    """Global settings for line segmentation"""

    margin: MarginSettings = field(default_factory=MarginSettings)
    main_text: MainTextSettings = field(default_factory=MainTextSettings)
    root_dir: Path = field(
        default_factory=lambda: Path(__file__)
        .resolve()
        .parent.parent.parent.parent.parent
    )

    def __post_init__(self):
        """Initialize paths after root_dir is set"""
        self.input_dir = self.root_dir / "data" / "processed" / "images"
        self.model_path = self.root_dir / "models" / "blla.mlmodel"

    def get_params(self, is_margin: bool) -> Dict[str, Any]:
        """Get parameters based on text type"""
        settings = self.margin if is_margin else self.main_text
        return {
            "is_margin": is_margin,
            "padding": settings.padding,
            "iou_threshold": settings.iou_threshold,
            "y_threshold": settings.y_threshold,
        }
