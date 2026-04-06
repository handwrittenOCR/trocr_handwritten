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
| 7 | Investigate discrepancies between act count and crop count | DONE |
| 8 | Improve the regex performance | In progress |
| 9 | Identify structure of manually transcribed data from Martinique | DONE |
| 10 | Run NER comparison (regex vs Flash vs Flash-Lite) on corrected dataset | pending |
| 11 | Launch NER (regex + LLM) on all communes | pending |

## Completed: investigate act vs crop discrepancies

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

## In progress: regexes on birth, deces, marriage records

## Completed: Martinique variable dictionary

Dataset: `C:\Users\marie\Dropbox\...\NER_datasets\raw\Martinique_manually_transcribed.csv`

**Objective**: merge OCR'd Guadeloupe acts with manually transcribed Martinique acts. Target schema: ours (per-type tables), with two additions from Martinique.

### Mapping: Martinique CSV → NER schema

| Martinique | NER field | Notes |
|---|---|---|
| `commune` | `commune` | |
| `year` | from `act_id` | |
| `type` (N/D/M) | `act_type` (naissance/deces/mariage) | |
| `registre_date` | `declaration_date` | Date act was registered |
| `event_date` | `birth_date` / `death_date` / `marriage_date` | Date of event (date only, no time) |
| `p_nom` | `child_name` / `person_name` / `spouse1_name` | |
| `p_age` | `person_age` (death/marriage only) | Birth acts: child is newborn, no age |
| `p_mat_let` | `registration_register` | Register letter |
| `p_mat` | `registration_number` | |
| `p_met` | `person_occupation` / `spouse1_occupation` | |
| `mere_nom` | `mother_name` | |
| `mere_age` | `mother_age` | |
| `mere_mat` | `mother_registration_number` | |
| `mere_met` | `mother_occupation` | |
| `mere_hab` | `habitation_name` | Act-level in our schema |
| `prop_nom` | `owner_name` | |
| `prop_com` | `owner_commune` | Added to our schema |
| `prop_demeure` | `owner_residence` | Added to our schema |
| `decl_nom` | `declarant_name` | |
| `epoux_nom` | `spouse1_name` | |
| `epouse_nom` | `spouse2_name` | |
| `epoux_age` | `spouse1_age` | |
| `epouse_age` | `spouse2_age` | |

### Gaps: Martinique has, we do not extract
- `p_surnom`, `surnom_mere`, `epouse_surnom` — nicknames
- `plantation_id` / `plantid_uniquenumber` — cross-registry linkage (Martinique-specific)

### Gaps: we extract, Martinique does not have
- `declaration_time`, `birth_time`, `death_time`
- `officer_name`
- `marge_act_number`, `marge_act_name`, `marge_act_owner`
- `father_*` fields

### Schema decisions
- `child_age` and `child_occupation` removed from `BirthActEntity` (newborns have no age)
- `owner_commune` and `owner_residence` added to all three entity classes
- Per-type tables preferred over Martinique's flat mixed table
