import json

from trocr_handwritten.utils.annotation import (
    SPLITS,
    assign_split,
    page_key,
    regroup_by_page,
)


class TestPageKey:
    """Tests for the page_key helper."""

    def test_strips_crop_index(self):
        assert (
            page_key("FRANOM22_COLH78_0261_0238_5.jpg") == "FRANOM22_COLH78_0261_0238"
        )

    def test_handles_dafcaom_naming(self):
        assert (
            page_key("0001_DAFCAOM04_DPPCEC_850116_0029_005")
            == "0001_DAFCAOM04_DPPCEC_850116_0029"
        )

    def test_crops_of_same_page_share_key(self):
        a = page_key("FRANOM58_078MIOM0839_0266_3.jpg")
        b = page_key("FRANOM58_078MIOM0839_0266_11.jpg")
        assert a == b

    def test_no_trailing_index_returns_stem(self):
        assert page_key("000.jpg") == "000"


class TestAssignSplit:
    """Tests for the page-aware assign_split."""

    def test_reuses_split_of_same_page(self):
        annotations = [{"filename": "FRANOM22_COLH78_0261_0238_0.jpg", "split": "test"}]
        result = assign_split("FRANOM22_COLH78_0261_0238_9.jpg", annotations)
        assert result == "test"

    def test_falls_back_to_random_without_context(self):
        assert assign_split() in SPLITS


def _make_crop(base, split, subfolder, stem, text):
    img = base / split / "images" / subfolder / f"{stem}.jpg"
    lbl = base / split / "labels" / subfolder / f"{stem}.txt"
    img.parent.mkdir(parents=True, exist_ok=True)
    lbl.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"fake")
    lbl.write_text(text, encoding="utf-8")


class TestRegroupByPage:
    """Tests for regroup_by_page."""

    def test_removes_page_leakage(self, tmp_path):
        _make_crop(tmp_path, "train", "Plein Texte", "DOC_0001_0", "a")
        _make_crop(tmp_path, "test", "Marge", "DOC_0001_1", "b")
        _make_crop(tmp_path, "dev", "Nom", "DOC_0001_2", "c")

        regroup_by_page(tmp_path)

        splits = set()
        for split in SPLITS:
            for img in (tmp_path / split / "images").rglob("*.jpg"):
                if page_key(img.name) == "DOC_0001":
                    splits.add(split)
        assert len(splits) == 1

    def test_annotations_consistent_with_disk(self, tmp_path):
        for i in range(20):
            _make_crop(tmp_path, "train", "Plein Texte", f"DOC_{i:04d}_0", "x")

        regroup_by_page(tmp_path)

        for split in SPLITS:
            ann_path = tmp_path / split / "annotations.json"
            if not ann_path.exists():
                continue
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            disk = list((tmp_path / split / "images").rglob("*.jpg"))
            assert len(ann) == len(disk)

    def test_drops_duplicate_copies(self, tmp_path):
        _make_crop(tmp_path, "train", "Nom", "DOC_0001_0", "same")
        _make_crop(tmp_path, "dev", "Nom", "DOC_0001_0", "same")

        regroup_by_page(tmp_path)

        copies = [
            img
            for split in SPLITS
            for img in (tmp_path / split / "images").rglob("DOC_0001_0.jpg")
        ]
        assert len(copies) == 1

    def test_idempotent(self, tmp_path):
        for i in range(15):
            _make_crop(tmp_path, "train", "Plein Texte", f"DOC_{i:04d}_0", "x")

        first = regroup_by_page(tmp_path)
        second = regroup_by_page(tmp_path)
        assert first == second
