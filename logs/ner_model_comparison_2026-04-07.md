# NER Model Comparison: Regex vs gemini-3-flash-preview vs gemini-3.1-flash-lite-preview

Date: 2026-04-07 11:36

Sample: 5 acts from abymes/1842

## Summary (Flash vs Flash-Lite)

| Metric | Count |
|---|---|
| Fields compared | 95 |
| Match (flash = flash-lite) | 64 |
| Mismatch | 10 |
| Flash only | 11 |
| Flash-Lite only | 0 |
| Both null | 10 |
| **Agreement rate** | **86.5%** |

## Per-field Agreement (Flash vs Flash-Lite)

| Field | Match | Mismatch | Flash only | Lite only | Both null |
|---|---|---|---|---|---|
| commune | 5 | 0 | 0 | 0 | 0 |
| death_date | 3 | 2 | 0 | 0 | 0 |
| death_place | 2 | 2 | 0 | 0 | 1 |
| declarant_age | 2 | 0 | 1 | 0 | 2 |
| declarant_name | 5 | 0 | 0 | 0 | 0 |
| declarant_occupation | 4 | 0 | 1 | 0 | 0 |
| declaration_date | 1 | 3 | 1 | 0 | 0 |
| habitation_name | 3 | 2 | 0 | 0 | 0 |
| officer_name | 4 | 0 | 1 | 0 | 0 |
| owner_commune | 0 | 0 | 3 | 0 | 2 |
| owner_name | 5 | 0 | 0 | 0 | 0 |
| owner_residence | 0 | 0 | 2 | 0 | 3 |
| person_age | 5 | 0 | 0 | 0 | 0 |
| person_name | 5 | 0 | 0 | 0 | 0 |
| person_occupation | 1 | 0 | 2 | 0 | 2 |
| person_qualifier | 4 | 1 | 0 | 0 | 0 |
| person_registration_number | 5 | 0 | 0 | 0 | 0 |
| person_registration_register | 5 | 0 | 0 | 0 | 0 |
| person_sex | 5 | 0 | 0 | 0 | 0 |

## Mismatches Detail

| Act | Field | Regex | Flash | Flash-Lite |
|---|---|---|---|---|
| FRAD971_1E35_055_115_010_C_order2 | person_qualifier | - | mulatresse | mulâtresse |
| FRAD971_1E35_055_115_010_C_order2 | declaration_date | jour mai | Dimanche troisième jour du mois de mai milhuit cent quarante et un | Dimanche troisième jour du mois de mai 1840 |
| FRAD971_1E35_055_115_010_C_order2 | habitation_name | - | habitation William Darasse | son habitation |
| 1839_FRAD971_1E35_074_118_061_C_order9 | death_date | - | ce jour | vingt neuf novembre 1839 |
| 1846_FRAD971_1E35_018_102_025_C_order8 | death_place | - | null | habitation Boisnormand |
| 1846_FRAD971_1E35_018_102_025_C_order8 | habitation_name | - | null | habitation Boisnormand |
| 1846_FRAD971_1E35_122_128_024_C_order4 | declaration_date | L'an mil huit cent quarante six, le trente un août | L'an mil huit cent quarante six, le trente un du mois d'août, à midi | le trente un du mois d'août mil huit cent quarante six |
| FRAD971_1E35_007_101_028_C_order4 | death_date | ce jour | douze du mois d'octobre mil huit cens quarante sept | 12 octobre 1847 |
| FRAD971_1E35_007_101_028_C_order4 | death_place | cette dernière | une des cases de l'habitation de cette dernière | habitation de la Demoiselle Marie Madelaine Popote |
| FRAD971_1E35_007_101_028_C_order4 | declaration_date | L'An mil huit cens quarante sept et le Mardi, douze octobre | Mardi, douze du mois d'octobre mil huit cens quarante sept | 12 octobre 1847 |

---

## Per-Act Detail

### lamentin_1840_FRAD971_1E35_055_115_010_C_order2 (deces)

**Marge:**
```
N° 19.
Décès d'Adol=
phine habiton
Vam Darasse.
```

**Plein Texte:**
```
L'An milhuit cent quarante et le Dimanche troisième
jour du mois de mai, à neuf heures du matin, Pardevant nous
Paul Madeleine Mathurin Brefford, maire de la commune du
Lamentin Ile Guadeloupe, est comparu en cette maison com=
mune, le Sieur William Darasse, âgé de quarante ans, propri
étaire domiciliée dans cette commune, lequel nous a déclaré que
hier à onze heures du matin, son esclave nommée Adolphine, mula
tresse âgée de deux ans, portée sur son dénombrement, est décédée
sur son habitation. D'après cette déclaration, nous avons
constaté le décès, avons autorisé l'inhumation et avons dressé
le présent acte que le déclarant a signé avec nous après lecture
```

| Field | Regex | Flash | Flash-Lite | Status |
|---|---|---|---|---|
| person_name | nommée Adolphine, mula tresse | Adolphine | Adolphine | = |
| person_sex | - | femme | femme | = |
| person_qualifier | - | mulatresse | mulâtresse | **!=** |
| person_age | 2 | 2 | 2 | = |
| person_occupation | - | - | - | - |
| person_registration_register | - | null | null | = |
| person_registration_number | - | null | null | = |
| death_date | hier | hier à onze heures du matin | hier | = |
| death_place | - | son habitation | son habitation | = |
| declaration_date | jour mai | Dimanche troisième jour du mois de mai milhuit cent quarante et un | Dimanche troisième jour du mois de mai 1840 | **!=** |
| declarant_name | - | William Darasse | William Darasse | = |
| declarant_age | - | 40 | 40 | = |
| declarant_occupation | - | propriétaire | propriétaire | = |
| owner_name | - | William Darasse | William Darasse | = |
| habitation_name | - | habitation William Darasse | son habitation | **!=** |
| owner_commune | - | Lamentin | - | flash> |
| owner_residence | - | Lamentin | - | flash> |
| officer_name | Paul Madeleine Mathurin Brefford | Paul Madeleine Mathurin Brefford | Paul Madeleine Mathurin Brefford | = |
| commune | - | Lamentin | Lamentin | = |

### petit_bourg_1839_FRAD971_1E35_074_118_061_C_order9 (deces)

**Marge:**
```
N° 134
décès
d'une femme
```

**Plein Texte:**
```
L'an mil huit cent trente neuf, et le vingt
neuf novembre, pardevant Nous Maire de la commune
du petit bourg, a comparu monsieur Clement Courau,
gereur de l'habitation la Retraite, sise en cette commune,
appartenant à monsieur Petra, lequel nous a déclaré que ce
jour, la négresse nommée Rosette, agée de cinquante deux
ans, attachée à la culture de ce bien, est décédée. Dont act
que le déclarant a signé avec nous. /.
```

| Field | Regex | Flash | Flash-Lite | Status |
|---|---|---|---|---|
| person_name | nommée Rosette | Rosette | Rosette | = |
| person_sex | femme | femme | femme | = |
| person_qualifier | - | négresse | négresse | = |
| person_age | 52 | 52 | 52 | = |
| person_occupation | - | attachée à la culture | - | flash> |
| person_registration_register | - | null | null | = |
| person_registration_number | - | null | null | = |
| death_date | - | ce jour | vingt neuf novembre 1839 | **!=** |
| death_place | Retraite | - | - | - |
| declaration_date | - | vingt neuf novembre mil huit cent trente neuf | - | flash> |
| declarant_name | - | Clement Courau | Clement Courau | = |
| declarant_age | - | - | - | - |
| declarant_occupation | - | gereur de l'habitation la Retraite | gereur de l'habitation la Retraite | = |
| owner_name | Petra | Petra | Petra | = |
| habitation_name | Retraite | la Retraite | la Retraite | = |
| owner_commune | - | - | - | - |
| owner_residence | - | - | - | - |
| officer_name | - | null | - | flash> |
| commune | - | petit bourg | petit bourg | = |

### anse_bertrand_1846_FRAD971_1E35_018_102_025_C_order8 (deces)

**Marge:**
```
Décès de Pompone
à M. Boisnormand & Sœurs
74
```

**Plein Texte:**
```
L'an mil huit cent quarante six, le vingtième jour du mois
d'Octobre, à midi, Nous Louis de Bébian Maire Officier de l'état
civil de la commune de l'anse Bertrand, avons reçu par lettre
en date de ce jour qui sera annexée au présent, signée par le Sr
Boisnormand propriétaire domicilié en cette commune, la
déclaration que la nommée Pompone de couleur noire, âgée
de quarante deux ans, inscrite sous le No 1721 du matricule,
appartenant à M. Boisnormand et sœurs, est décédée hier
dix neuf du courant à onze heures du matin. En foi de
quoi le présent acte et avons signé.
```

| Field | Regex | Flash | Flash-Lite | Status |
|---|---|---|---|---|
| person_name | Pompone | Pompone | Pompone | = |
| person_sex | femme | femme | femme | = |
| person_qualifier | - | noire | noire | = |
| person_age | 42 | 42 | 42 | = |
| person_occupation | - | null | - | flash> |
| person_registration_register | - | null | null | = |
| person_registration_number | 1721 | 1721 | 1721 | = |
| death_date | la nommée Pompone de couleur noire, âgée de quarante deux ans, inscrite sous le No 1721 du matricule, appartenant | dix neuf du courant à onze heures du matin | dix neuf du courant | = |
| death_place | - | null | habitation Boisnormand | **!=** |
| declaration_date | L'an mil huit cent quarante six, le vingtième jour Octobre | vingtième jour du mois d'Octobre mil huit cent quarante six | vingtième jour du mois d'Octobre mil huit cent quarante six | = |
| declarant_name | - | Boisnormand | Boisnormand | = |
| declarant_age | - | null | - | flash> |
| declarant_occupation | - | propriétaire | - | flash> |
| owner_name | M. Boisnormand et sœurs | M. Boisnormand et sœurs | Boisnormand et sœurs | = |
| habitation_name | - | null | habitation Boisnormand | **!=** |
| owner_commune | - | l'anse Bertrand | - | flash> |
| owner_residence | - | null | - | flash> |
| officer_name | Louis de Bébian | Louis de Bébian | Louis de Bébian | = |
| commune | l | l'anse Bertrand | anse Bertrand | = |

### sainte_anne_1846_FRAD971_1E35_122_128_024_C_order4 (deces)

**Marge:**
```
N° 162.
Décès
Immiègre
```

**Plein Texte:**
```
L'an mil huit cent quarante six, le trente un du mois d'août, à
midi, nous Le Terrier d'Eschinardville, Soussigné, Maire de la
Commune de Sainte Anne grand terre Guadeloupe, certifions avoir
reçu par lettre, signée par Mr Bourgoin, daté de ce jour, habitant
propriétaire, domicilié en cette Commune, la déclaration que son nègre
nommé Désiré, âgé de quarante huit ans, tonnelier, immatriculé
sous le N° 6840, est décédé sur son habitation, le jour d'hier, à
cinq heures du matin. En foi de quoi nous avons signé le présent
acte.
```

| Field | Regex | Flash | Flash-Lite | Status |
|---|---|---|---|---|
| person_name | nommé Désiré | Désiré | Désiré | = |
| person_sex | homme | homme | homme | = |
| person_qualifier | - | nègre | nègre | = |
| person_age | 48 | 48 | 48 | = |
| person_occupation | tonnelier | tonnelier | tonnelier | = |
| person_registration_register | - | null | null | = |
| person_registration_number | 6840 | 6840 | 6840 | = |
| death_date | son nègre nommé Désiré, âgé de quarante huit ans, tonnelier, immatriculé sous le N° 6840, est décédé sur son habitation, le jour d'hier | le jour d'hier, à cinq heures du matin | le jour d'hier | = |
| death_place | - | son habitation | son habitation | = |
| declaration_date | L'an mil huit cent quarante six, le trente un août | L'an mil huit cent quarante six, le trente un du mois d'août, à midi | le trente un du mois d'août mil huit cent quarante six | **!=** |
| declarant_name | Bourgoin | Mr Bourgoin | Mr Bourgoin | = |
| declarant_age | - | - | - | - |
| declarant_occupation | - | habitant propriétaire | habitant propriétaire | = |
| owner_name | Bourgoin | Mr Bourgoin | Mr Bourgoin | = |
| habitation_name | - | son habitation | son habitation | = |
| owner_commune | - | - | - | - |
| owner_residence | - | - | - | - |
| officer_name | Le Terrier d'Eschinardville | Le Terrier d'Eschinardville | Le Terrier d'Eschinardville | = |
| commune | sainte | Sainte Anne grand terre Guadeloupe | Sainte Anne grand terre Guadeloupe | = |

### abymes_1847_FRAD971_1E35_007_101_028_C_order4 (deces)

**Marge:**
```
Augustin, fils de Marie
appt. à la Dlle Popote
décédé le 12. 8bre 1847
N: 79
```

**Plein Texte:**
```
L'An mil huit cens quarante sept et le Mardi, douze du mois d'octobre,
à deux heures et demie de l'après midi. Pardevant nous François Elie David,
Maire de la commune des Abymes, ile grand'Terre Guadeloupe, remplissant
les fonctions d'officier de l'état civil de la dite commune, est comparu le Sieur
Victor Théotime, âgé de quarante quatre ans, maçon domicilié à la Pointe-
à-Pitre, lequel nous a déclaré que ce jour, à deux heures du matin, l'esclave
Augustin, négrillon d'un an, fils de Marie, immatriculé registre T. n° 1499 et appartenant
à la Demoiselle Marie Madelaine Popote, est décédé dans une des cases de
l'habitation de cette dernière, sise en la commune des Abymes, et le déclarant a signé
avec nous le présent acte après lecture.
```

| Field | Regex | Flash | Flash-Lite | Status |
|---|---|---|---|---|
| person_name | - | Augustin | Augustin | = |
| person_sex | - | homme | homme | = |
| person_qualifier | - | négrillon | négrillon | = |
| person_age | - | 1 | 1 | = |
| person_occupation | - | - | - | - |
| person_registration_register | T | T | T | = |
| person_registration_number | 1499 | 1499 | 1499 | = |
| death_date | ce jour | douze du mois d'octobre mil huit cens quarante sept | 12 octobre 1847 | **!=** |
| death_place | cette dernière | une des cases de l'habitation de cette dernière | habitation de la Demoiselle Marie Madelaine Popote | **!=** |
| declaration_date | L'An mil huit cens quarante sept et le Mardi, douze octobre | Mardi, douze du mois d'octobre mil huit cens quarante sept | 12 octobre 1847 | **!=** |
| declarant_name | - | Victor Théotime | Victor Théotime | = |
| declarant_age | - | 44 | 44 | = |
| declarant_occupation | - | maçon | maçon | = |
| owner_name | Marie Madelaine Popote | Marie Madelaine Popote | Marie Madelaine Popote | = |
| habitation_name | cette dernière | habitation de la Demoiselle Marie Madelaine Popote | habitation de la Demoiselle Marie Madelaine Popote | = |
| owner_commune | - | Les Abymes | - | flash> |
| owner_residence | - | - | - | - |
| officer_name | François Elie David | François Elie David | François Elie David | = |
| commune | abymes | Les Abymes | Les Abymes | = |
