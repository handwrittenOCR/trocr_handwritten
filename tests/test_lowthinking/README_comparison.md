# OCR Model Comparison — Ground Truth Evaluation

## Purpose

Compare OCR transcription quality between **HIGH thinking** (old, uncapped) and **LOW thinking** (new, cost-optimized) runs of Gemini 3.1 Pro Preview on cropped civil registry regions from Abymes (1841).

## Folder structure

```
test_lowthinking/
├── images/          # Source crop JPGs (what the model sees)
├── old/             # Transcriptions from HIGH thinking (default, uncapped)
├── new/             # Transcriptions from LOW thinking (cost-optimized)
├── comparison.md    # Side-by-side raw output comparison
├── comparison_template.csv   # <-- RA fills this in
└── README_comparison.md      # This file
```

## Instructions for the RA

### Step 1 — Read the image

Open the crop image from `images/`. These are cropped regions from 19th-century French civil registries (slaves' registries from Guadeloupe, 1841).

### Step 2 — Fill in ground truth

Open `comparison_template.csv` in Excel/Google Sheets. For each row, look at the image and fill in the **ground truth** columns:

| Column | Description | Example |
|--------|-------------|---------|
| `type_acte` | Type of civil act | Naissance, Deces, Mariage |
| `numero_acte` | Act number as written in margin | 1, 2, 3... |
| `nom` | Family name of the subject (if any) | -- (enslaved persons often have none) |
| `prenom` | First name of the subject | Collette, Colombe, Vite |
| `sexe` | Sex | M, F |
| `couleur` | Color/race as written | noir/noire, mulatre/mulatresse |
| `age` | Age of the subject | 28 ans |
| `nom_mere` | Mother's name | Juliette, Anne Marie |
| `age_mere` | Mother's age | 28 ans |
| `nom_proprietaire` | Name of the owner | La Marochelle, Nafrechoux |
| `qualite_proprietaire` | Owner's title/quality | habitant proprietaire |
| `commune` | Parish/commune | Abymes |
| `date_evenement` | Date of the event (birth, death...) | 30 Decembre 1840 |
| `date_declaration` | Date the act was declared | 4 Janvier 1841 |
| `declarant` | Person who declared the act | Chs Catalogne |
| `immatricule` | Registration number | R. N 3003 |
| `notes_ra` | Any notes (illegible parts, damage...) | Bottom half water damaged |
| `ground_truth_full` | Full transcription as read by RA | (free text) |

**Important:**
- Write `[illisible]` for parts you cannot read
- Write `NA` if the field does not apply (e.g., Marge crops may not have all fields)
- For Marge regions, typically only: type_acte, numero_acte, prenom, couleur, nom_mere, age_mere, nom_proprietaire, date_evenement, immatricule
- For Plein Texte regions, typically all fields are present

### Step 3 — Compare model outputs

Read the corresponding `.md` files in `old/` (HIGH thinking) and `new/` (LOW thinking). Copy the full text into:
- `old_high_thinking` column
- `new_low_thinking` column

### Step 4 — Evaluate accuracy

For each row, mark in the match columns:
- `old_match`: How well does the OLD transcription match ground truth? Score: `exact`, `minor_errors`, `major_errors`, `wrong`
- `new_match`: Same for NEW transcription

### Key variables to evaluate

Priority variables for accuracy comparison (in order of importance):

1. **prenom** — Name of the enslaved person
2. **nom_mere** — Mother's name (key for genealogy)
3. **nom_proprietaire** — Owner's name
4. **date_evenement** — Date of event
5. **type_acte** — Act type (Naissance/Deces/Mariage)
6. **immatricule** — Registration number
7. **age / age_mere** — Ages
8. **couleur** — Color designation

## Sample data

18 crops total from 4 pages of Abymes 1841:
- 9 Plein Texte crops (full civil act text)
- 9 Marge crops (margin annotations with key metadata)

## Cost comparison

| Mode | Total crops | Thinking tokens | Total cost |
|------|------------|-----------------|------------|
| HIGH (old) | ~2,482 across full abymes | uncapped | ~$700 total bill |
| LOW (new, round 1) | 6 | 986 (1 crop) | EUR 0.029 |
| LOW (new, round 2) | 12 | 0 | EUR 0.036 |
