# LOW vs HIGH Thinking — OCR Comparison

**Model:** gemini-3-pro-preview (redirects to gemini-3.1-pro-preview)
**Thinking mode (new):** LOW  |  **Thinking mode (old):** HIGH (default, uncapped)

---

## Plein Texte — `FRAD971_1E35_001_101_002_C` — crop `000`

**Tokens:** input=1218, output=184, thinking=0

### OLD (HIGH thinking)
```
Le présent régistre contenant cinquante neuf feuillets, non
compris celui-ci, a été coté et paraphé, sur chaque feuillet, par Nous,
Joseph Dupuy Desislets Mondésir, Officier de la légion d'honneur, juge
Royal du Tribunal de première instance de l'arrondissement de
la Pointe à Pitre, pour servir à l'enrégistrement des actes de
Naissances, Mariages et [Décès?] des esclaves qui auront
lieu pendant l'année Mil [huit?] [cent?] quarante et un, à l'état
civil de la Commune des Abimes, conformément aux
articles 17 & 18 de l'ordonn[ance?] Royale du 11 juin 1839
Donné en notre Hôtel, à la Pointe-à-Pitre,
Guadeloupe, le cinq novembre mil huit cent quarante
[Dupuy Mondésir?]
```

### NEW (LOW thinking)
```
Le présent registre contenant cinquante neuf feuillets, non
compris celui-ci, a été coté et paraphé, sur chaque feuillet, par Nous,
Joseph Dupuy Desislets Mondésir, Officier de la légion d'honneur, juge
Royal du tribunal de première instance de l'arrondissement de
la Pointe à Pitre, pour servir à l'enregistrement des actes de
Naissances, Mariages et [illisible] des esclaves qui auront
lieu pendant l'année Mil huit cent quarante et un, à l'état
civil de la Commune des Abimes, conformément aux
articles 17 & 18 de l'ordonnance Royale du 11 juin 1839
Donné en notre Hôtel, à la Pointe-à-Pitre,
Guadeloupe, le cinq novembre mil huit cent quarante
```

---

## Plein Texte — `FRAD971_1E35_001_101_003_C` — crop `003`

**Tokens:** input=1238, output=155, thinking=0

### OLD (HIGH thinking)
```
L'an mil huit Cent quarante un le Lundi quatre
du mois de Janvier à Cinq heures du Soir, Pardevant nous adjoint
faisant fonction de Maire, officier de l'Etat Civil de la Commune
des Abymes, Ile Grande Terre, Guadeloupe; est Comparu le
Sieur Chs Catalogne agé de Vingt deux ans, habitant
propriétaire lequel nous a déclaré que la négresse nommée
Rose agée de Vingt huit ans, appartenant au Sieur
Nafrechoux père, habitant propriétaire domicilié en cette
Commune est accouchée le Trente Décembre dernier d'une fille
noire qui a eu nom Colombe et a Signé avec nous après lecture
Chs Catalogne
```

### NEW (LOW thinking)
```
L'an mil huit Cent quarante un le Lundi quatre
du mois de Janvier à Cinq heures du Soir, Pardevant nous adjoint
faisant fonctions de Maire officier de l'Etat Civil de la Commune
des Abymes, Ile Grande Terre, Guadeloupe, est Comparu le
Sieur Chs Catalogne agé de Vingt deux ans, habitant
propriétaire lequel nous a déclaré que la negresse nommée
Rose agée de Vingt huit ans, appartenant au Sieur
Saferchaux père, habitant propriétaire domicilié en cette
Commune est accouchée le Trente Décembre dernier d'une fille
noire qui a eu nom Colombe et a Signé avec nous après lecture
```

---

## Plein Texte — `FRAD971_1E35_001_101_003_C` — crop `000`

**Tokens:** input=1218, output=160, thinking=986

### OLD (HIGH thinking)
```
l'an mil huit cent quarante un le quatre
du mois de Janvier à six heures du soir pardevant nous
faisant fonction de Maire et officier de l'Etat Civil de la
Commune des Abymes, île Grande Terre, Guadeloupe
est comparu le sieur Ch. Catalogne agé de vingt deux
ans, habitant propriétaire, lequel nous a déclaré que la
négresse Anne Marie agée de trente ans appartenant
au sieur Laferchau père, habitant propriétaire audit
en cette Commune est accouchée hier seize du courant à
six heures du matin d'un garçon noir qui a eu nom Cité
et a signé après lecture. Ch. Catalogne
```

### NEW (LOW thinking)
```
L'an mil huit Cent quarante un et le Quatorze
du mois de Janvier à dix heures du Soir Pardevant nous
faisant fonction de Maire et officier de l'Etat Civil de la
Commune des abymes, Ile Grande Terre, Guadeloupe
est Comparu le Sieur Ch. Catalogne agé de Vingt deux
ans, habitant propriétaire, lequel nous a declaré que la
negresse Anne Marie agée de Trente ans appartenant
au Sieur Laperchau pere habitant propriétaire domicilié
en cette Commune est accouchée hier treize du Courant à
dix heures du matin d'un garçon noir qui a eu nom Tite
et a signé avec nous après lecture. Ch. Catalogne
```

---

## Marge — `FRAD971_1E35_001_101_003_C` — crop `001`

**Tokens:** input=1219, output=78, thinking=0

### OLD (HIGH thinking)
```
Naissance 1. [illisible]
de la noire Collette
fille de la négresse
Juliette de 28 ans
appart. au Sieur La
Marochelle père,
habitant Proprietaire
aux abymes, née le
30 Décembre 1840.

Immatricule R.
N° 3003

[illisible]
```

### NEW (LOW thinking)
```
Naissance 1. 1
de la noire Céleste
fille de la négresse
illatre de 22 ans
appart. au sieur la
Nofrecheu fils
habitant domicilié
aux abymes, née le
30 décembre 1840.

Immatricule R.
N° 3003
```

---

## Marge — `FRAD971_1E35_001_101_003_C` — crop `004`

**Tokens:** input=1226, output=12, thinking=0

### OLD (HIGH thinking)
```
GUADELOUPE
MAIRIE
ABYMES
```

### NEW (LOW thinking)
```
GUADELOUPE
MAIRIE
ABYMES
```

---

## Marge — `FRAD971_1E35_001_101_003_C` — crop `005`

**Tokens:** input=1218, output=83, thinking=0

### OLD (HIGH thinking)
```
Naissance N° 2.
nommé Vite, fils de la
mineure Anne Marie
de Hans appartenant
au sieur Marlenheim
né le [illisible]
au polygone [illisible]
Fevrier 1841 à [illisible]
heures du matin

Immatriculé Registre
D N° 3434 -
```

### NEW (LOW thinking)
```
Quittance N° 2.
du mois d'oût, fils de la
majeure Anne Marie
de Hans, appartenant
au sieur Molsheim
[illisible]
[illisible]
[illisible] 1841 à
[illisible]
[illisible]
Immatriculé Registre
D N° 3434 -
```

---


## Cost Summary

```
Model: gemini-3-pro-preview
Total calls: 6
Input tokens:           7,337  (EUR 0.0124)
Output tokens:            672  (EUR 0.0068)
Thinking tokens:          986  (EUR 0.0100)
--------------------------------------------------
TOTAL ESTIMATED COST: EUR 0.0293
WARNING: 986 thinking tokens detected! Verify your Google billing matches this estimate.
```

# Round 2 — Pages 004_C and 005_C

## Plein Texte -- `FRAD971_1E35_001_101_004_C` -- crop `000`

**Tokens:** input=1230, output=189, thinking=0

### OLD (HIGH thinking)
```
L'an mil huit Cent [soixante et un?] le Trente [et un?]
Janvier à dix heures du matin. Pardevant nous adjoint
faisant fonction de Maire et officier de l'Etat Civil
de la Commune [des Abymes, île?] de la Grande Terre
Guadeloupe, est comparu le Sieur Ch. Catalogne,
âgé de [cinquante?] ans, propriétaire, lequel
nous a déclaré qu'une petite fille sans vie
est née en son habitation de la dame [Adélaïde?] habitante
domiciliée en cette Commune, le Trente
un du mois de Decembre Dernier à Cinq heures du Soir
[et auquel?] il a donné le nom [Félicité?] et a Signé
avec nous après lecture. - Ch. Catalogne,
```

### NEW (LOW thinking)
```
L'an mil huit Cent quarante un le Jeudi Sept
Janvier à Sept heures du matin. Pardevant nous adjoint
faisant fonction de Maire et officier de l'Etat Civil
de la Commune de [illisible] Ile Grande Terre
Guadeloupe, est Comparu le Sieur Ch. Catalogne
âgé de [illisible] ans, [illisible] de [illisible], lequel
nous a déclaré qu'une négresse [illisible] de [illisible]
[illisible] appartenant à la dame V. [illisible] habitante
[illisible] de [illisible] est décédée le Mardi trente
un du mois de Décembre Dernier à Cinq heures du Soir
[illisible] qu'il a [illisible] nom [illisible] et a Signé
avec nous après lecture. Ch. Catalogne,
```

---

## Plein Texte -- `FRAD971_1E35_001_101_004_C` -- crop `004`

**Tokens:** input=1218, output=179, thinking=0

### OLD (HIGH thinking)
```
L'an mil huit Cent quarante un et le [vingt huit?]
du mois de [novembre?] à [dix?] heures du matin Par devant nous
[illisible] maire faisant fonctions d'officier de l'état
civil de la Commune d'[Aigrefeuille?] Canton [du dit?]
lieu arrondissement de [Rochefort?] (Charente Inférieure) Est
comparu le Sieur [Guillaud?] âgé de [trente?] ans
profession de [cultivateur?] domicilié à [illisible] lequel nous a déclaré que la
nommée [Marie?] âgée de [vingt?] ans, appartenant
à la dite Commune, est accouchée le [jour d'hier?] à
[cinq?] heures du soir d'un garçon
auquel il a donné les prénoms de [Jean Baptiste?]
```

### NEW (LOW thinking)
```
Naissance de M. L. L'an mil huit Cent quarante un et le Vingt un
du mois de decembre a dix heures du matin par devant
nous maire faisant fonctions d'officier de l'Etat
Civil de la Commune de [illisible] Canton de
[illisible] arrondissement de [illisible]
[illisible] est comparu le Sieur
[illisible] agé de [illisible] ans
[illisible] lequel nous a declaré que la
nommée [illisible] agée de [illisible] ans, appartenant
a la dite [illisible]
domiciliée en cette Commune, est accouchée le Vingt
du present mois a Cinq heures du soir d'un garcon
[illisible] a eu nom [illisible] et a signé avec nous
```

---

## Marge -- `FRAD971_1E35_001_101_004_C` -- crop `001`

**Tokens:** input=1211, output=51, thinking=0

### OLD (HIGH thinking)
```
Liasse n° 8.
de mes [titres?]
et papiers
le tout en un
Sac mis
a part [illisible]
pour etre remis
a Mr Dampierre
[illisible]
```

### NEW (LOW thinking)
```
Liasse n° 8.
de nos [illisible]
et [illisible]
[illisible]
[illisible]
[illisible]
[illisible]
[illisible]
[illisible]
```

---

## Marge -- `FRAD971_1E35_001_101_004_C` -- crop `003`

**Tokens:** input=1254, output=4, thinking=0

### OLD (HIGH thinking)
```
Ch.. Catalogne
```

### NEW (LOW thinking)
```
Ch. Catalogne
```

---

## Plein Texte -- `FRAD971_1E35_001_101_005_C` -- crop `001`

**Tokens:** input=1251, output=48, thinking=0

### OLD (HIGH thinking)
```
L'an mil huit cent quarante un et le [neuf?]
neuf de Janvier à neuf heures du matin
[illisible]
[illisible]
[illisible]
```

### NEW (LOW thinking)
```
L'an mil huit Cent quarante un et le dix
Neuf de Janvier à neuf heures du matin
[illisible]
[illisible]
[illisible]
[illisible]
[illisible]
```

---

## Plein Texte -- `FRAD971_1E35_001_101_005_C` -- crop `002`

**Tokens:** input=1230, output=133, thinking=0

### OLD (HIGH thinking)
```
L'an mil huit cent quarante deux le vingt huit Fevrier
à huit heures du matin Pardevant nous adjoint faisant fonctions d'
officier de l'Etat Civil de la Commune de
Port-Louis île grande terre Guadeloupe est comparu
le Sieur Joseph [Durand?] agé de
cinquante cinq ans, gérant de l'habitation
dite Beauport en cette Commune, lequel nous
a déclaré que le nommé Antoine agé de
cinquante ans environ appartenant à la dite habitation est
décédé hier soir sur les onze heures et
le déclarant a signé avec nous après lecture
[illisible]
```

### NEW (LOW thinking)
```
L'an mil huit cent cinquante deux le dix huit Fevrier
à huit heures du matin Par devant nous adjoint faisant fonctions de Maire
officier de l'Etat Civil de la Commune de
Basse-Terre, est comparu le Sieur Joseph Duport agé de
cinquante cinq ans, gérant de l'habitation
de la dite Dame Commune, lequel nous
a declaré le decès du nommé Antoine agé de
cinquante ans, appartenant à la dite habitation et
décédé hier soir du Courant à trois heures de
l'après midi et a signé avec nous après lecture.
```

---

## Plein Texte -- `FRAD971_1E35_001_101_005_C` -- crop `003`

**Tokens:** input=1211, output=163, thinking=0

### OLD (HIGH thinking)
```
L'an mil huit Cent [illisible] le onze may
a dix heures du matin par devant nous maire faisant fonction
d'officier de l'Etat Civil de la Commune
de St. Louis Canton de Grandebourg est comparu
le Sieur Pierre Bte Desrivieres agé de quarante
Cinq ans habitant domicilié en Cette
Commune, lequel nous a declaré que la fille de nom
Joseph agée de Vingt quatre ans appartenant
a la dame Veuve Desrivieres sa mere est accouchée
dans Cette Commune d'un Enfant Male
nommé Omer et a le declarans requis de ce signé
avec nous après lecture faite
                                              Pte Desrivieres
         [illisible] Maire
                                              [illisible]
```

### NEW (LOW thinking)
```
L'an mil huit Cent quarante un le onze may a
dix heures du matin par devant nous adjoint faisant fonction
de Maire et officier de l'Etat Civil de la Commune
de St. françois Canton de la Guadeloupe est
Comparu le sieur pte. Desravines agé de quarante
Cinq ans habitant domicilié en Cette
Commune, lequel nous a déclaré que le fils du sieur
Joseph agé de Vingt quatre ans appartenant
à l'habitation de M. de Labaume sa résidence
en Cette Commune est décédé hier
à dix heures du soir dans la maison de son
maitre après lecture il a signé
pte. Desravines
[illisible]
[illisible]
[illisible]
```

---

## Plein Texte -- `FRAD971_1E35_001_101_005_C` -- crop `007`

**Tokens:** input=1226, output=95, thinking=0

### OLD (HIGH thinking)
```
L'an mil huit Cinquante un
le vingt trois juin à [huit?] heures du
Pardevant nous maire faisant fonctions
d'officier de l'Etat Civil de la
Commune de Ste Suzanne Arrond.
d'Orthez Departement des Basses
Pyrénées est comparu le sieur Cazenave
Jean agé de Cinquante sept ans
cultivateur domicilié à Ste Suzanne
Lequel nous a declaré que
cejourd'hui à onze heures
du matin Marie Cazenave
```

### NEW (LOW thinking)
```
L'an mil huit Cinquante un
[illisible]
Pardevant nous maire faisant fonction
d'officier de l'Etat Civil de la
Commune de [illisible]
Canton de [illisible]
[illisible]
[illisible]
[illisible] Cinquante [illisible]
[illisible]
[illisible]
[illisible] lecture. - [illisible]
```

---

## Marge -- `FRAD971_1E35_001_101_005_C` -- crop `004`

**Tokens:** input=1238, output=65, thinking=0

### OLD (HIGH thinking)
```
Vingt [illisible]
de la [illisible]
Claude [illisible]
appartenant [illisible]
et [illisible]
du [illisible]
[illisible]
[illisible]
[illisible]
[illisible]
2304
[illisible]
```

### NEW (LOW thinking)
```
[illisible]
de la negresse
Clairine [illisible]
appart. a M.
de [illisible]
du [illisible]
[illisible]
[illisible]
[illisible]
[illisible]
2304
[illisible]
```

---

## Marge -- `FRAD971_1E35_001_101_005_C` -- crop `005`

**Tokens:** input=1223, output=56, thinking=0

### OLD (HIGH thinking)
```
Décès N° 3.
de la négresse Sevienne
de 76 ans appartenant
au Sr. Chevalier [illisible]
habitant de cette [illisible]
[illisible]
[illisible]
[illisible]
[illisible]
```

### NEW (LOW thinking)
```
Décés N° 3.
de la négresse Sevranne
de 76 ans appartent.
au s.r Chassaigne [illisible]
[illisible]
[illisible]
[illisible]
[illisible]
```

---

## Marge -- `FRAD971_1E35_001_101_005_C` -- crop `008`

**Tokens:** input=1230, output=18, thinking=0

### OLD (HIGH thinking)
```
J. Cas
[Delaporte?]
8bre 11.
```

### NEW (LOW thinking)
```
[illisible]
[illisible]
8006 [illisible]
```

---

## Marge -- `FRAD971_1E35_001_101_005_C` -- crop `009`

**Tokens:** input=1218, output=29, thinking=0

### OLD (HIGH thinking)
```
Debet M. le [illisible]
du mois Mars a
A pour effets fournis
a la dame sa fille
habits et linges
```

### NEW (LOW thinking)
```
Décès de la
nommée Jeanne a
l'age de 40 ans
[illisible]
[illisible]
```

---


## Round 2 Cost Summary

```
Model: gemini-3-pro-preview
Total calls: 12
Input tokens:          14,740  (EUR 0.0250)
Output tokens:          1,030  (EUR 0.0105)
Thinking tokens:            0  (EUR 0.0000)
--------------------------------------------------
TOTAL ESTIMATED COST: EUR 0.0355
```
