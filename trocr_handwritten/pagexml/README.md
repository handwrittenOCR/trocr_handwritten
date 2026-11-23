# PAGE-XML corpus: manifest and consistency check

The corpus is one PAGE-XML file (schema PAGE 2019-07-15) per page. Each region carries, in its `custom` attribute:

```
layout {class:Marge;} handwriting {status:manuscrit;} transcription {status:transcrit; split:train; crop:<crop>; category:Marge;} provenance {box:detecteur; validated:true;}
```

- `layout`: one of the 8 classes (Title, Bruit, Marge, Nom, Plein Texte, Signature, Table, Section);
- `handwriting`: `manuscrit` / `non_manuscrit`;
- `transcription`: `transcrit` (with the HTR crop: split, crop name, category, and the text in `TextEquiv`), `non_transcrit`, or `rejete` (handwritten, reading uncertain, kept out of the HTR dataset);
- `provenance`: origin of the box (`box:`), and `validated:true` once checked by hand.

The PAGE-XML files are the primary data. `pages_manifest.csv` is a table derived from them, one row per region.

## Rebuild the manifest and check the corpus

```bash
python -m trocr_handwritten.pagexml.manifest --out data/processed/pages_manifest.csv
```

Reads `data/processed/page_xml/`, takes each page's split from the detection dataset (`data/layout_merged/<split>/annotations.json`) and carries `ls_key` (internal Label Studio key) over from the existing manifest (`--ls-keys`). Then checks the corpus against the HTR dataset (`--htr-dir`, default `data/ocr/20260807-ft-data`):

- every page has a split and every box is marked validated;
- a region has a text if and only if it is transcribed;
- every transcribed region has its crop (image, label, entry of `annotations.json`) in the split of its page, with the same text;
- every HTR crop belongs to a transcribed region.

It prints the problems found and exits with code 1 if there is any. Schema validation is separate: `xmllint --noout --schema pagecontent.xsd data/processed/page_xml/*.xml`.
