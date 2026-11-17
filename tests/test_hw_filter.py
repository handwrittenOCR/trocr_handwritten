import numpy as np

from trocr_handwritten.hw_filter.features import measures, square
from trocr_handwritten.hw_filter.settings import CLASSES, HwFilterSettings
from trocr_handwritten.hw_filter.train import scores


class TestHwFilterSettings:
    """Tests for HwFilterSettings."""

    def test_default_values(self):
        """Default settings are the retained filter."""
        s = HwFilterSettings()
        assert s.embed_model == "facebook/dinov2-small"
        assert s.image_size == 448
        assert s.C == 0.1
        assert s.threshold == 0.5
        assert s.classes == CLASSES


class TestFeatures:
    """Tests for the crop measures and the square padding."""

    def test_measures_blank_vs_ink(self):
        """A crop with strokes has more ink and components than a blank one."""
        blank = np.full((60, 200), 230, np.uint8)
        ink = blank.copy()
        ink[20:40, 20:180:10] = 20
        mb, mi = measures(blank), measures(ink)
        assert mb["ink"] == 0
        assert mi["ink"] > 0
        assert mi["cc_n"] > mb["cc_n"]

    def test_square_shape_and_empty(self):
        """Square returns an RGB image of the requested side, even for an empty crop."""
        im = square(np.full((30, 90), 200, np.uint8), 64)
        assert im.size == (64, 64) and im.mode == "RGB"
        assert square(np.zeros((0, 0), np.uint8), 32).size == (32, 32)


class TestScores:
    """Tests for the evaluation scores."""

    def test_scores_per_class(self):
        """Accuracy and non manuscrit F1, overall and per class."""
        y = np.array([1, 0, 1, 0])
        p = np.array([1, 0, 0, 0])
        cls = np.array(["Marge", "Marge", "Section", "Section"])
        s = scores(y, p, cls, ["Marge", "Section", "Nom"])
        assert s["acc"] == 0.75
        assert s["acc_Marge"] == 1.0
        assert s["acc_Section"] == 0.5
        assert "acc_Nom" not in s
        assert s["r_non_manuscrit"] == 1.0
