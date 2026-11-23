import json

from trocr_handwritten.pagexml.manifest import build_rows, check
from trocr_handwritten.pagexml.page import field, read_corpus

PAGE = """<?xml version='1.0' encoding='UTF-8'?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageFilename="p1.jpg" imageWidth="100" imageHeight="100">
    <TextRegion id="r1" type="marginalia" custom="layout {class:Marge;} handwriting {status:manuscrit;} transcription {status:transcrit; split:train; crop:p1_000; category:Marge;} provenance {box:detecteur; validated:true;}">
      <Coords points="10,20 40,20 40,60 10,60"/>
      <TextEquiv><Unicode>Né le 3 mai</Unicode></TextEquiv>
    </TextRegion>
    <TextRegion id="r2" type="other" custom="layout {class:Bruit;} handwriting {status:non_manuscrit;} transcription {status:non_transcrit;} provenance {box:manuelle;}">
      <Coords points="0,0 5,0 5,5 0,5"/>
    </TextRegion>
  </Page>
</PcGts>"""


def corpus(tmp_path, text="Né le 3 mai"):
    """A one-page corpus and its HTR dataset (one Marge crop with ``text``) under tmp_path."""
    (tmp_path / "xml").mkdir()
    (tmp_path / "xml" / "p1.xml").write_text(PAGE, encoding="utf-8")
    for s in ("train", "dev", "test"):
        (tmp_path / "htr" / s).mkdir(parents=True)
        entries = []
        if s == "train":
            for sub, name, data in (
                ("images", "p1_000.jpg", "x"),
                ("labels", "p1_000.txt", text),
            ):
                (tmp_path / "htr" / s / sub / "Marge").mkdir(parents=True)
                (tmp_path / "htr" / s / sub / "Marge" / name).write_text(
                    data, encoding="utf-8"
                )
            entries = [
                {
                    "filename": "p1_000.jpg",
                    "subfolder": "Marge",
                    "text": text,
                    "split": "train",
                }
            ]
        (tmp_path / "htr" / s / "annotations.json").write_text(
            json.dumps(entries), encoding="utf-8"
        )
    return read_corpus(tmp_path / "xml")


class TestField:
    """Tests for reading the custom attribute."""

    def test_blocks_and_keys(self):
        """Each block is read separately; a missing key gives None."""
        c = "handwriting {status:manuscrit;} transcription {status:transcrit; split:dev; crop:a_1;} provenance {box:detecteur;}"
        assert field(c, "handwriting") == "manuscrit"
        assert field(c, "transcription") == "transcrit"
        assert field(c, "transcription", "split") == "dev"
        assert field(c, "provenance", "box") == "detecteur"
        assert field(c, "provenance", "validated") is None
        assert field(c, "layout", "class") is None


class TestManifest:
    """Tests for the manifest rows and the consistency check."""

    def test_rows(self, tmp_path):
        """One row per region, with box, statuses, crop and carried-over ls_key."""
        rows = build_rows(
            corpus(tmp_path), {"p1.jpg": "train"}, {"r1": "lot/Marge/r1.jpg"}
        )
        assert [r["region_id"] for r in rows] == ["r1", "r2"]
        assert rows[0] == {
            "page": "p1.jpg",
            "page_xml": "page_xml/p1.xml",
            "split": "train",
            "region_id": "r1",
            "class": "Marge",
            "handwriting": "manuscrit",
            "transcription": "transcrit",
            "htr_category": "Marge",
            "htr_crop": "p1_000",
            "ls_key": "lot/Marge/r1.jpg",
            "x": 10,
            "y": 20,
            "w": 30,
            "h": 40,
            "box_provenance": "detecteur",
        }
        assert rows[1]["htr_crop"] == "" and rows[1]["ls_key"] == ""

    def test_check_flags_unvalidated_box(self, tmp_path):
        """A consistent corpus only reports the box without validated:true."""
        assert check(corpus(tmp_path), {"p1.jpg": "train"}, tmp_path / "htr") == [
            "p1.jpg r2: box not marked validated"
        ]

    def test_check_flags_text_and_split(self, tmp_path):
        """A text differing from the HTR label and a crop outside the page split are reported."""
        c = corpus(tmp_path, text="Né le 8 mai")
        assert "p1.jpg r1: text differs between PAGE-XML and HTR" in check(
            c, {"p1.jpg": "train"}, tmp_path / "htr"
        )
        problems = check(c, {"p1.jpg": "dev"}, tmp_path / "htr")
        assert "p1.jpg r1: crop split train != page split dev" in problems
