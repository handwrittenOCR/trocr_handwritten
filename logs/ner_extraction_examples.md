# NER Extraction Examples — Abymes 1842

Side-by-side comparison of Marge, Regex, and LLM (Gemini 3 Flash) extractions for the first 5 acts in the dataset.

---

## Act 1: Registry Header (unknown type)

### Source

**Marge:** *(none)*

**Plein Texte:**
> Le présent régistre contenant quarante six feuillets, non compris celui ci, a été coté et paraphé, sur chaque feuillet, par Nous, Joseph Dupuy Désislets Mondésir, Officier de la légion d'honneur, Juge Royal du tribunal de première instance de l'arrondissement de la Pointe-à-Pitre, pour servir à l'enregistrement des actes de Naissances, Mariages et Décès des Esclaves qui auront lieu pendant l'année Mil huit cent quarante deux à l'état Civil de la Commune des Abimes, conformément aux articles 17 et 18 de l'ordonnance Royale du 11 juin 1839.

### Extracted

| Field | Marge | Regex | LLM |
|---|---|---|---|
| officer_name | — | — | Joseph Dupuy Désislets Mondésir |
| commune | — | — | Abimes |

> **Note:** The registry header is not a civil act — it's the opening page. Only the LLM extracts the judge's name and commune. The regex finds no structured entities.

---

## Act 2: Décès N° 1 — angelle

### Source

**Marge:**
> Décès N° 1
> de angelle agée de 27 ans
> ang Sre Duluc & totile

**Plein Texte:**
> L'An Mil huit cent quarante deux et le samedi premier Janvier à onze heures du matin. Pardevant nous Maire et officier de l'état civil de la commune des abymes, ile grande terre Guadeloupe. Est comparu le Sieur Louis Leo Armaignac agé de trente deux ans habitant propriétaire domicilié en la commune des abymes, Lequel nous a déclaré que ce matin à une heure et sur l'habitation cocotiers appartenant aux sieurs Estete et Duluc, la négresse angelle agée de vingt sept ans immatriculée Registre D. n° 3076, y est décédée, et a signé avec nous le dit acte après lecture faite.

### Extracted

| Field | Marge | Regex | LLM |
|---|---|---|---|
| person_name | angelle | angelle | angelle |
| person_sex | — | femme | femme |
| person_age | 27 ans | 27 | 27 |
| person_registration_register | — | D | D |
| person_registration_number | — | 3076 | 3076 |
| death_date | — | ce matin | 01/01/1842 |
| death_time | — | une heure | 01:00 |
| death_place | — | cocotiers | habitation cocotiers |
| declaration_date | — | — | 01/01/1842 |
| declaration_time | — | — | 11:00 |
| habitation_name | — | cocotiers | cocotiers |
| owner_name | Sre Duluc & totile | Estete et Duluc | Estete et Duluc |
| declarant_name | — | Louis Leo Armaignac | Louis Leo Armaignac |
| declarant_age | — | 32 | 32 |
| declarant_occupation | — | habitant propriétaire | habitant propriétaire |
| commune | — | abymes | Les Abymes |

> Both regex and LLM agree on all core fields. The LLM resolves dates to absolute format. Owner correctly extracted from "appartenant aux sieurs Estete et Duluc" in the Plein Texte.

---

## Act 3: Décès N° 2 — Caraïse

### Source

**Marge:**
> Décès N° 2
> de Laraise agé de 1 an
> au Sr Boucaud

**Plein Texte:**
> L'An Mil huit cent quarante deux et le lundi trois du mois de Janvier à sept heures du matin. Pardevant nous Maire et officier de l'état civil de la commune des abymes, de grande terre Guadeloupe. Est comparu le Sieur Louis hilaire Zephores Boricaud agé de trente sept ans habitant propriétaire domicilié en la commune des abymes, Lequel nous a déclaré que ce matin à une heure et sur l'habitation la Baronie de cette Commune, le petit nègre Caraïse agé de un an, immatriculé Registre D N° 3328 - y est décédé, et a signé avec nous le dit acte après lecture faite

### Extracted

| Field | Marge | Regex | LLM |
|---|---|---|---|
| person_name | Laraise | Caraïse | Caraïse |
| person_sex | — | homme | homme |
| person_age | 1 an | 1 | 1 |
| person_registration_register | — | D | D |
| person_registration_number | — | 3328 | 3328 |
| death_date | — | ce matin | 3 janvier 1842 |
| death_time | — | une heure | 01:00 |
| death_place | — | Baronie | Habitation la Baronie |
| declaration_date | — | trois Janvier | 3 janvier 1842 |
| declaration_time | — | sept heures du matin | 07:00 |
| habitation_name | — | Baronie | la Baronie |
| owner_name | au Sr Boucaud | — | **Sr Boucaud** |
| declarant_name | — | Louis hilaire Zephores Boricaud | Louis hilaire Zephores Boricaud |
| declarant_age | — | 37 | 37 |
| declarant_occupation | — | habitant propriétaire | habitant propriétaire |
| commune | — | abymes | Les Abymes |

> **Key finding:** The Plein Texte says "l'habitation la Baronie de cette Commune" with no "appartenant" — the regex cannot identify the owner. The Marge says "au Sr Boucaud" — the LLM uses this to correctly extract the owner. The Marge also shows a different spelling of the name ("Laraise" vs "Caraïse").

---

## Act 4: Naissance N° 1 — Noel

### Source

**Marge:**
> Naissance 16° 1
> de Noel enfant de françoise
> bon Longval

**Plein Texte:**
> L'An Mil huit cent quarante Deux et le mercredi cinq du mois de Janvier à neuf heures du matin. Pardevant Nous Maire et officier de l'état civil de la Commune des abymes, de Grande terre Guadeloupe. EST Comparu le sieur Isaac Blanchet Dubelloy agé de quarante cinq ans, habitant propriétaire domicilié en la commune des abymes, lequel nous a déclaré que ce matin à quatre heures et dans une des cases de l'habitation Longval de cette commune appartenant à Monsieur Vernias la negresse francoise agée de vingt trois ans immatriculée Registre C N° 2303 est accouchée d'un enfant noir qui a eu nom Noel, et immatriculé Registre D N° 3494 et a signé avec nous le dit acte après lecture faite.

### Extracted

| Field | Marge | Regex | LLM |
|---|---|---|---|
| **Child** | | | |
| child_name | Noel | Noel | Noel |
| child_sex | — | — | **homme** |
| child_registration_register | — | D | D |
| child_registration_number | — | 3494 | 3494 |
| **Mother** | | | |
| mother_name | françoise | francoise | francoise |
| mother_sex | — | femme | femme |
| mother_age | — | 23 | vingt trois ans |
| mother_registration_register | — | C | C |
| mother_registration_number | — | 2303 | 2303 |
| **Act details** | | | |
| birth_date | — | ce matin | 5 Janvier 1842 |
| birth_time | — | quatre heures | quatre heures du matin |
| habitation_name | bon Longval | Longval | Longval |
| owner_name | — | Vernias | Monsieur Vernias |
| declarant_name | — | Isaac Blanchet Dubelloy | Isaac Blanchet Dubelloy |
| declarant_age | — | 45 | quarante cinq ans |
| declarant_occupation | — | habitant propriétaire | habitant propriétaire |
| declaration_date | — | cinq Janvier | 5 Janvier 1842 |
| declaration_time | — | neuf heures du matin | neuf heures du matin |
| commune | — | abymes | les abymes |

> The LLM infers child sex as **homme** from the masculine form "immatriculé" (vs "immatriculée" for female). The Marge says "bon Longval" — "bon" is an abbreviation for the habitation. Owner correctly extracted from "appartenant à Monsieur Vernias" by both methods.

---

## Act 5: Naissance N° 4 — Françoise

### Source

**Marge:**
> Naissance N.o 4
> de francoise enf.te de Marie
> Therese au S.r Duburand

**Plein Texte:**
> L'An Mil huit cent quarante deux et le samedi quinze du mois de janvier à neuf heures du matin. Pardevant nous Maire et officier de l'état civil de la commune des abymes, ile Grande terre Guadeloupe. Est comparu le Sieur Georges Marie Joseph Fontaine Dubérand agé de cinquante ans, habitant propriétaire domicilié en la commune des abymes, Lequel nous a déclaré que ce matin a une heure et dans une des cases de son habitation dite dorginy sa négresse Marie Thérèse agée de trente neuf ans, immatriculée Registre N°          y est accouchée d'un enfant noir du sexe féminin, qui a eu nom Françoise, immatriculée Registre D N° 3517 et a signé avec nous le dit acte après lecture faite.

### Extracted

| Field | Marge | Regex | LLM |
|---|---|---|---|
| **Child** | | | |
| child_name | francoise | Françoise | Françoise |
| child_sex | — | femme | femme |
| child_registration_register | — | D | D |
| child_registration_number | — | 3517 | 3517 |
| **Mother** | | | |
| mother_name | Marie Therese | Marie Thérèse | Marie Thérèse |
| mother_sex | — | femme | femme |
| mother_age | — | 39 | trente neuf ans |
| mother_registration_register | — | N | — |
| mother_registration_number | — | — | — |
| **Act details** | | | |
| birth_date | — | ce matin | 15 janvier 1842 |
| birth_time | — | une heure | une heure |
| habitation_name | — | dorginy | dorginy |
| owner_name | au S.r Duburand | Georges Marie Joseph Fontaine Dubérand | Georges Marie Joseph Fontaine Dubérand |
| declarant_name | — | Georges Marie Joseph Fontaine Dubérand | Georges Marie Joseph Fontaine Dubérand |
| declarant_age | — | 50 | cinquante ans |
| declarant_occupation | — | habitant propriétaire | habitant propriétaire |
| declaration_date | — | quinze janvier | 15 janvier 1842 |
| declaration_time | — | neuf heures du matin | neuf heures du matin |
| commune | — | abymes | Les Abymes |

> **Key finding:** "son habitation" and "sa négresse" clearly indicate the declarant is the owner. Both regex and LLM correctly extract the declarant as owner. The Marge confirms: "au S.r Duburand". Note the mother's registration is incomplete in the OCR: "Registre N°          " (missing data in the original document).

---

## Summary

### What each source contributes

| Source | Strengths | Weaknesses |
|---|---|---|
| **Marge** | Owner name (abbreviated), person name, act type/number | Abbreviated, OCR errors, no registration numbers |
| **Regex** | Free, fast, deterministic, good on structured fields (registration, declarant) | Misses implicit ownership, relative dates unresolved |
| **LLM** | Resolves dates, infers sex from grammar, uses Marge for owner | ~$0.001/act, may hallucinate if prompt not strict |

### Owner extraction comparison (updated prompt)

| Act | Marge | Regex | LLM | Source of owner info |
|---|---|---|---|---|
| Décès N°1 (angelle) | Sre Duluc | Estete et Duluc | Estete et Duluc | Plein Texte: "appartenant aux sieurs" |
| Décès N°2 (Caraïse) | au Sr Boucaud | — | **Sr Boucaud** | Marge: "au Sr Boucaud" |
| Naissance N°1 (Noel) | — | Vernias | Monsieur Vernias | Plein Texte: "appartenant à Monsieur" |
| Naissance N°4 (Françoise) | au S.r Duburand | Dubérand | Dubérand | Plein Texte: "son habitation", "sa négresse" |
