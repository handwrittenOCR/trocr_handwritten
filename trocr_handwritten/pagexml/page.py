import re
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
Q = "{%s}" % NS


def field(custom, block, key="status"):
    """
    Value of ``key`` inside a block of a PAGE ``custom`` attribute.

    Args:
        custom: Attribute text, e.g. ``"layout {class:Marge;} handwriting {status:manuscrit;}"``.
        block: Block name (layout, handwriting, transcription, provenance).
        key: Key inside the block.

    Returns:
        str or None: The value, None if the block or the key is absent.
    """
    m = re.search(rf"{block} \{{([^}}]*)\}}", custom or "")
    if not m:
        return None
    v = re.search(rf"(?:^|;)\s*{key}:([^;]*);", m.group(1))
    return v.group(1).strip() if v else None


def box_of(el):
    """
    Bounding box of a region element.

    Args:
        el: PAGE region element with a ``Coords`` child.

    Returns:
        list: ``[x, y, w, h]`` in pixels.
    """
    pts = [
        tuple(map(int, p.split(",")))
        for p in el.find(f"{Q}Coords").get("points").split()
    ]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def read_regions(path):
    """
    Read the regions of one PAGE-XML file of the corpus.

    Args:
        path: PAGE-XML file.

    Returns:
        tuple: (image filename, list of region dicts with region_id, class, handwriting,
        transcription, htr_split, htr_category, htr_crop, box, box_provenance, validated, text),
        regions in document order.
    """
    page = ET.parse(path).getroot().find(f"{Q}Page")
    regions = []
    for el in page:
        if not el.tag.endswith("Region"):
            continue
        c = el.get("custom", "")
        u = el.find(f"{Q}TextEquiv/{Q}Unicode")
        regions.append(
            {
                "region_id": el.get("id"),
                "class": field(c, "layout", "class"),
                "handwriting": field(c, "handwriting"),
                "transcription": field(c, "transcription"),
                "htr_split": field(c, "transcription", "split"),
                "htr_category": field(c, "transcription", "category"),
                "htr_crop": field(c, "transcription", "crop"),
                "box": box_of(el),
                "box_provenance": field(c, "provenance", "box"),
                "validated": field(c, "provenance", "validated") == "true",
                "text": u.text if u is not None else None,
            }
        )
    return page.get("imageFilename"), regions


def read_corpus(page_xml_dir):
    """
    Read every PAGE-XML file of a folder, sorted by file name.

    Args:
        page_xml_dir: Folder of ``<page>.xml`` files.

    Returns:
        list: (xml path, image filename, regions) per page.
    """
    return [(p, *read_regions(p)) for p in sorted(Path(page_xml_dir).glob("*.xml"))]
