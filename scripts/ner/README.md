# NER Extraction Tasks

## Status

| # | Task | Status |
|---|------|--------|
| 1 | Display N random margin/plein texte pairs to verify matching | DONE |
| 2 | Move act_type detection OUT of dataset.py into NER step | DONE |
| 3 | Add regex extraction of act_type from Marge | DONE |
| 4 | Add regex extraction of act_type from Plein Texte | DONE |
| 5 | Add regex extraction of marge_act_name, marge_act_owner, marge_act_number | DONE |
| 6 | Fix "Aujourd'hui" preamble variant not recognized as new act | DONE |
| 7 | Investigate discrepancies between act count and crop count | pending |
| 8 | Run NER comparison (regex vs Flash vs Flash-Lite) on corrected dataset | pending |
| 9 | Launch NER (regex + LLM) on all communes | pending |

## Next: investigate act vs crop discrepancies

Dataset rebuilt: **9,293 acts** from 65 commune/year combinations.
Some communes show large gaps between crop count and act count (many continuations merged).
Suspicious cases to investigate:
- deshaies/1843: 115 crops -> 11 acts (extreme collapse)
- sainte_anne/1840: 240 crops -> 157 acts
- abymes/1841: 171 crops -> 107 acts
- petit_bourg/1838: 241 crops -> 176 acts

Possible causes: missing preamble variants, OCR errors in preamble text, legitimate multi-crop acts.

## Architecture decision

**dataset.py** should only build the raw dataset: pair Marge with Plein Texte, handle continuations, output one record per act. No extraction logic.

**act_type detection** is an NER task, done separately:
- From Marge: regex on margin text (deces/naissance/mariage keywords)
- From Plein Texte: regex on body text (death/birth indicators)
- Both results stored; Marge-based is more reliable when available

## Context

- Dataset: 9,293 acts across 8 communes (rebuilt 2026-04-06 after Aujourd'hui fix)
- Reading order corrected: `metadata_reading_order.json` (1,668 pages)
- Sort order fixed: left page first, then right page (was interleaved by y-center)
- Regex extractor now extracts from both Marge (type, name, owner, number) and Plein Texte
- LLM NER sends both Marge + Plein Texte together
