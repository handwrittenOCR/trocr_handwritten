# Model Comparison — Gemini 3.x on Handwritten Civil Registries

Date: 2026-03-24
Test data: Crops from Abymes 1842 (pages 003 and 004)

## Speed & Cost Summary

| Model | Avg/image | Cost (input/M) | Cost (output/M) |
|-------|-----------|----------------|-----------------|
| gemini-3-flash-preview | 13s (but up to 259s!) | $0.50 | $3.00 |
| gemini-3-pro-preview | 16.8s | $2.00 | $12.00 |
| gemini-3.1-flash-lite-preview | **5.7s** | **$0.15** | **$0.60** |
| gemini-3.1-pro-preview | 18.8s | $2.50 | $15.00 |

---

## Full Transcriptions — Plein Texte/003.jpg (Décès Caraïse)

### gemini-3-flash-preview (13.0s)
```
L'An Mil huit cent quarante deux et le lundi trois du mois de
Janvier à sept heures du matin. Pardevant nous Maire et officier de
l'état civil de la commune des abymes, de grande terre Guadeloupe. Est
comparu le Sieur Louis hilaire Zephores Boricaud agé de trente sept ans
habitant propriétaire domicilié en la commune des abymes, Lequel nous a
déclaré que ce matin à une heure et sur l'habitation la Baronie de cette
Commune, le petit nègre Caraïse agé de un an, immatriculé Registre D N°
3328 - y est décédé, et a signé avec nous le dit acte après lecture faite
```

### gemini-3-pro-preview (18.9s)
```
L'An Mil huit cent quarante deux et le lundi trois du mois de
Janvier à sept heures du matin. Pardevant nous Maire et officier de
l'état civil de la commune des abymes, ile grande terre Guadeloupe. Est
Comparu le Sieur Louis hilaire Zephores Boricaud agé de trente sept ans
habitant propriétaire domicilié en la commune des abymes, Lequel nous a
declaré que ce matin à une heure et sur l'habitation la Baronie de cette
Commune, le petit nègre Caraïse agé de un an, immatriculé Registre D N°
3328 - y est décédé, et a signé avec nous le dit acte après lecture faite
```

### gemini-3.1-flash-lite-preview (2.5s)
```
L'An Mil huit cent quarante deux et le lundi trois du mois de
Janvier à sept heures du matin. Pardevant nous Maire et officier de
l'état civil de la commune des abymes, de grande terre Guadeloupe. Est
comparu le sieur Louis hilaire [Zephore?] Boricaud agé de trente sept ans
habitant propriétaire domicilié en la commune des abymes, lequel nous a
declaré que ce matin à une heure et sur l'habitation la Baronie de cette
commune, le petit negre Caraise agé de un an, immatriculé Registre D n°
3328. y est décédé, et a signé avec nous le dit acte après lecture faite
```

### gemini-3.1-pro-preview (18.8s)
```
L'An Mil huit cent quarante deux et le lundi trois du mois de
Janvier à sept heures du matin. Pardevant nous Maire et officier de
l'état civil de la commune des abymes, ile grande terre Guadeloupe. Est
Comparu le Sieur Louis hilaire Zephores Boricaud agé de trente sept ans
habitant propriétaire domicilié en la commune des abymes, Lequel nous a
declaré que ce matin à une heure et sur l'habitation la Baronie de cette
Commune, le petit nègre Caraïse agé de un an, immatriculé Registre D N°
3328 - y est décédé, et a signé avec nous le dit acte après lecture faite
```

---

## Full Transcriptions — Marge/000.jpg (Décès N°2)

### gemini-3-flash-preview (258.7s)
```
Décès N° 2
de Laraise agé de 1 an
au Sr Boucaud
```

### gemini-3-pro-preview (14.7s)
```
Déces N° 2
de Caraise agé de 1 ans
au Sr Boucaud
```

### gemini-3.1-flash-lite-preview (1.3s)
```
Déces N° 2
de Caraise agé de 1 an
au St Boucand
```

### gemini-3.1-pro-preview (15.9s)
```
Déces N° 2
de Caraise agé de 1 ans
au Sr Boucaud
```

---

## Full Transcriptions — Plein Texte/004.jpg (Naissance Noël)

### gemini-3.1-flash-lite-preview
```
L'An Mil huit cent quarante Deux et le mercredi cinq du mois de
Janvier à neuf heures du matin. Pardevant Nous Maire et officier de
l'état civil de la Commune des abymes, de Grande terre Guadeloupe. EST
Comparu le sieur Isaac Blanchet Dubelloy agé de quarante cinq ans,
habitant proprietaire domicilié en la commune des abymes, lequel nous a declaré
que ce matin à quatre heures et dans une des cases de l'habitation Longral de
cette commune appartenant à Monsieur Vernias la negresse francoise agé de
vingt trois ans immatriculée Registre C N° 2303 est accouchée d'un enfant noir
qui a eu nom Noel, et immatriculé Registre D N° 3494 et a signé avec nous le dit
acte après lecture faite.
```

### gemini-3-pro-preview
```
L'An Mil huit cent quarante Deux et le mercredi cinq du mois de
Janvier a' neuf heures du matin. Pardevant Nous Maire et officier de
l'etat civil de la Commune des abymes, de Grande terre Guadeloupe. EST
Comparu le sieur Isaac Blanchet Dubelloy agé de quarante cinq ans,
habitant proprietaire domicilié en la commune des abymes, lequel nous a declaré
que ce matin a' quatre heures et dans une des cases de l'habitation Longval de
cette commune appartenant a' Monsieur Vernias la negresse francoise agée de
Vingt trois ans immatriculée Registre C N° 2303 est accouchée d'un enfant noir
qui a eu nom Noel, et immatriculé Registre D N° 3494 et a signé avec nous le dit
acte après lecture faite.
```

---

## Full Transcriptions — Plein Texte/007.jpg (Décès Angelle)

### gemini-3.1-flash-lite-preview
```
L'An Mil huit cent quarante deux et le samedi premier Janvier à onze
heures du matin. Pardevant nous Maire et officier de l'état civil de la commune des
abymes, ile grande terre Guadeloupe. Est comparu le sieur Louis Leo Armaignac
agé de trente deux ans habitant propriétaire domicilié en la commune des abymes,
lequel nous a declaré que ce matin à une heure et sur l'habitation cocotiers
appartenant aux sieurs Estite et Duluc, la négresse angelle agée de vingt sept ans
immatriculée Registre D. n° 3046, y est décédée, et a signé avec nous le dit acte
après lecture faite.
```

### gemini-3-pro-preview
```
L'An Mil huit cent quarante deux et le samedi premier Janvier à onze
heures du matin. Pardevant nous Maire et officier de l'état civil de la commune des
abymes, ile grande terre Guadeloupe. Est comparu le sieur Louis Léo Armaignac
agé de trente deux ans habitant propriétaire domicilié en la commune des abymes,
Lequel nous a declaré que ce matin à une heure et sur l'habitation cocotiers
appartenant aux sieurs Estite et Duluc, la négresse angelle agée de vingt sept ans
immatriculée Registre D. n° 3076, y est décédée, et a signé avec nous le dit acte
après lecture faite.
```

---

## Full Transcriptions — Marge/001.jpg (Décès N°1)

### gemini-3.1-flash-lite-preview
```
Déces N° 1
de angelle agée de 27 ans
au Srs Duluc & totile
```

### gemini-3-pro-preview
```
Décès N° 1
de angelle agée de 27 ans
aux Srs Duluc & lotite
```

---

## Full Transcriptions — Marge/002.jpg (Naissance N°1)

### gemini-3.1-flash-lite-preview
```
Naissance 16° 1
de Noel enfant de françoise
bne Longval
```

### gemini-3-pro-preview
```
Naissance 16° 1
de Noel enfant de françoise
hon Longval
```

---

## Full Transcriptions — Page 004 / Plein Texte/003.jpg (Naissance Françoise Dubérand)

### gemini-3.1-flash-lite-preview
```
L'An Mil huit cent quarante deux et le samedi quinze du mois de
janvier à neuf heures du matin. Pardevant nous Maire et officier de l'état
civil de la commune des abymes, ile Grande terre Guadeloupe. Est comparu le
Sieur Georges Marie Joseph Fontaine Dubérand agé de cinquante ans,
habitant propriétaire domicilié en la commune des abymes, lequel nous a
déclaré que ce matin a une heure et dans une des cases de son habitation
dite doryny sa négresse Marie Thérèse agée de trente neuf ans, immatriculée
Registre N° y est accouchée d'un enfant noir du sexe feminin,
qui a eu nom Françoise, immatriculée Registre D N° 3517 et a signé avec
nous le dit acte après lecture faite.
```

### gemini-3-pro-preview
```
L'An Mil huit cent quarante deux et le samedi quinze du mois de
janvier à neuf heures du matin. Pardevant nous Maire et officier de l'état
civil de la commune des abymes, ile Grande terre Guadeloupe. Est Comparu le
Sieur Georges Marie Joseph Fontaine Dubérand agé de Cinquante ans,
habitant propriétaire domicilié en la commune des abymes, Lequel nous a
déclaré que ce matin a une heure et dans une des cases de son habitation
dite dorigny sa négresse Marie Thérèse agée de trente neuf ans, immatriculée
Registre N°     y est accouchée d'un enfant noir du sexe feminin,
qui a eu nom Françoise, immatriculée Registre D N° 3517 et a signé avec
nous le dit acte après lecture faite.
```

---

## Full Transcriptions — Page 004 / Marge/005.jpg (Décès N°3)

### gemini-3.1-flash-lite-preview
```
Décès N° 3
de Hilaire agé de 20 ans
s/on Longval
```

### gemini-3-pro-preview
```
Décès N° 3
de hilaire agé de 20 ans
hon Longval
```

---

## Key Differences Summary

| Detail | flash-lite | Pro | Impact |
|--------|-----------|-----|--------|
| `Zephores` | `[Zephore?]` (uncertain) | `Zephores` (confident) | Name accuracy |
| `Boucaud` | `Boucand` | `Boucaud` | Name accuracy |
| `Longval` | `Longral` | `Longval` | Place name accuracy |
| `n° 3076` | `n° 3046` | `n° 3076` | Registry number accuracy |
| `nègre` | `negre` | `nègre` | Accent preservation |
| `Caraïse` | `Caraise` | `Caraïse` | Diacritical accuracy |
| `Décès` | `Déces` | `Décès` | Accent preservation |
| `Laraise` (3-flash) | `Caraise` | `Caraise` | 3-flash misread the name entirely |
| Speed | **5.7s avg** | 16.8s avg | 3x faster |
| Cost/1M input | **$0.15** | $2.00 | 13x cheaper |

**Conclusion**: Pro models (3-pro and 3.1-pro) are significantly more accurate on names, registry numbers, and accents. Flash-lite is fast and cheap but produces errors on the details most critical for civil registry research.
