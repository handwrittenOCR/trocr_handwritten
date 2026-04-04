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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Type | Preamble | Preamble |
| Parish | Abymes | Abymes |
| Date | 5 novembre 1840 | 5 novembre 1840 |
| Note | Reads "Deces?" with ? | Marks as [illisible] |
| Signature | [Dupuy Mondesir?] | (omitted) |

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Act type | Naissance | Naissance |
| Slave name | Colombe | Colombe |
| Sex / color | F / noire | F / noire |
| Mother name | Rose | Rose |
| Mother age | 28 ans | 28 ans |
| **Owner name** | **Nafrechoux pere** | **Saferchaux pere** |
| Owner title | habitant proprietaire | habitant proprietaire |
| Parish | Abymes | Abymes |
| Event date | 30 Decembre 1840 | 30 Decembre 1840 |
| Declaration date | 4 Janvier 1841 | 4 Janvier 1841 |
| Declarant | Chs Catalogne | Chs Catalogne |

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Act type | Naissance | Naissance |
| **Slave name** | **Cite** | **Tite** |
| Sex / color | M / noir | M / noir |
| Mother name | Anne Marie | Anne Marie |
| Mother age | 30 ans | 30 ans |
| **Owner name** | **Laferchau pere** | **Laperchau pere** |
| Owner title | habitant proprietaire | habitant proprietaire |
| Parish | Abymes | Abymes |
| **Event date** | **hier 16 du courant** | **hier 13 du courant** |
| **Declaration date** | **4 Janvier 1841** | **14 Janvier 1841** |
| Declarant | Ch. Catalogne | Ch. Catalogne |

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Act type | Naissance | Naissance |
| Act number | 1 | 1 |
| **Slave name** | **Collette** | **Celeste** |
| Sex / color | F / noire | F / noire |
| **Mother name** | **Juliette** | **illatre** |
| **Mother age** | **28 ans** | **22 ans** |
| **Owner name** | **La Marochelle pere** | **la Nofrecheu fils** |
| Owner title | habitant proprietaire | habitant domicilie |
| Parish | Abymes | Abymes |
| Event date | 30 Decembre 1840 | 30 Decembre 1840 |
| Registration | R. N 3003 | R. N 3003 |

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Type | Header stamp | Header stamp |
| Parish | Abymes | Abymes |
| Note | Identical | Identical |

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| **Act type** | **Naissance** | **Quittance** |
| Act number | 2 | 2 |
| Slave name | Vite | Vite |
| Sex | M | M |
| Mother name | Anne Marie de Hans | Anne Marie de Hans |
| **Mother status** | **mineure** | **majeure** |
| **Owner name** | **Marlenheim** | **Molsheim** |
| Event date | Fevrier 1841 | 1841 |
| Registration | D N 3434 | D N 3434 |

---


## Cost Summary (Round 1)

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

---

# Round 2 — Pages 004_C and 005_C

## Plein Texte — `FRAD971_1E35_001_101_004_C` — crop `000`

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| **Act type** | **Stillbirth (nee sans vie)** | **Death (decedee)** |
| **Slave name** | **Felicite?** | **[illisible]** |
| Sex / color | F / noire | F |
| **Owner name** | **dame Adelaide?** | **dame V. [illisible]** |
| Plantation | habitation (owner's) | [illisible] |
| **Parish** | **Abymes?** | **[illisible]** |
| Event date | 31 Decembre | 31 Decembre |
| **Declaration date** | **31 Janvier** | **7 Janvier 1841** |
| **Year** | **soixante et un? (1861?)** | **quarante un (1841)** |
| Declarant | Ch. Catalogne | Ch. Catalogne |

---

## Plein Texte — `FRAD971_1E35_001_101_004_C` — crop `004`

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Act type | Naissance | Naissance |
| **Slave name** | **Jean Baptiste?** | **[illisible]** |
| Sex | M | M |
| **Mother name** | **Marie?** | **[illisible]** |
| **Mother age** | **20 ans?** | **[illisible]** |
| **Owner name** | **Guillaud?** | **[illisible]** |
| **Parish** | **Aigrefeuille? (Charente Inf.)** | **[illisible]** |
| **Declaration date** | **28 novembre?** | **21 decembre** |
| **Event date** | **jour d'hier** | **20 du present mois** |
| Declarant | Guillaud? | [illisible] |
| Note | OLD guesses many names with ? | NEW marks most as illisible |

---

## Marge — `FRAD971_1E35_001_101_004_C` — crop `001`

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Type | Administrative note | Administrative note |
| Note | OLD reads "titres, papiers, Mr Dampierre" | NEW marks most as illisible |

---

## Marge — `FRAD971_1E35_001_101_004_C` — crop `003`

**Tokens:** input=1254, output=4, thinking=0

### OLD (HIGH thinking)
```
Ch.. Catalogne
```

### NEW (LOW thinking)
```
Ch. Catalogne
```

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Type | Signature | Signature |
| Declarant | Ch.. Catalogne | Ch. Catalogne |
| Note | Minor: double dot in OLD | Identical content |

---

## Plein Texte — `FRAD971_1E35_001_101_005_C` — crop `001`

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| **Declaration date** | **neuf? neuf Janvier 1841** | **dix Neuf Janvier 1841** |
| Note | Mostly illegible | Mostly illegible |

---

## Plein Texte — `FRAD971_1E35_001_101_005_C` — crop `002`

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Act type | Death | Death |
| Slave name | Antoine | Antoine |
| Sex | M | M |
| Age | 50 ans | 50 ans |
| Plantation | habitation Beauport | habitation (unnamed) |
| **Parish** | **Port-Louis** | **Basse-Terre** |
| **Declarant** | **Joseph Durand?** | **Joseph Duport** |
| **Declaration date** | **28 Fevrier 1842** | **18 Fevrier 1852** |
| Event date | hier soir | hier soir |

---

## Plein Texte — `FRAD971_1E35_001_101_005_C` — crop `003`

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| **Act type** | **Birth (accouchee)** | **Death (decede)** |
| **Slave name** | **Omer** | **(unnamed)** |
| Sex | M | M |
| **Mother/Subject** | **Joseph (fille, 24 ans)** | **Joseph (fils, 24 ans)** |
| **Owner name** | **Vve Desrivieres** | **M. de Labaume** |
| Plantation | (unnamed) | habitation de Labaume |
| **Parish** | **St. Louis, Grandebourg** | **St. francois** |
| Declaration date | 11 mai | 11 mai |
| **Declarant** | **Pierre Bte Desrivieres** | **pte. Desravines** |

---

## Plein Texte — `FRAD971_1E35_001_101_005_C` — crop `007`

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Slave name | Marie Cazenave | [illisible] |
| **Parish** | **Ste Suzanne (Basses Pyrenees)** | **[illisible]** |
| **Declarant** | **Cazenave Jean, 57 ans** | **[illisible]** |
| Note | Appears to be French mainland act | NEW marks almost everything illisible |

---

## Marge — `FRAD971_1E35_001_101_005_C` — crop `004`

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| **Slave name** | **Claude** | **Clairine** |
| Color | [illisible] | negresse |
| Owner name | [illisible] | M. de [illisible] |
| Registration | 2304 | 2304 |
| Note | Heavily damaged crop | Heavily damaged crop |

---

## Marge — `FRAD971_1E35_001_101_005_C` — crop `005`

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Act type | Death | Death |
| Act number | 3 | 3 |
| **Slave name** | **Sevienne** | **Sevranne** |
| Sex / color | F / negresse | F / negresse |
| Age | 76 ans | 76 ans |
| **Owner name** | **Sr. Chevalier** | **s.r Chassaigne** |

---

## Marge — `FRAD971_1E35_001_101_005_C` — crop `008`

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Note | OLD reads "J. Cas / Delaporte / 8bre 11" | NEW reads "8006 [illisible]" |
| Note | Entirely different readings | Very short fragment |

---

## Marge — `FRAD971_1E35_001_101_005_C` — crop `009`

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

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| **Act type** | **Financial note (Debet)** | **Death (Deces)** |
| **Slave name** | **(none)** | **Jeanne** |
| **Age** | **NA** | **40 ans** |
| Note | Completely different interpretation | One reads commerce, the other a death |

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

# Round 3 — Page FRAD971_1E35_014_102_024_C (Anse-Bertrand 1842)

## Plein Texte — `FRAD971_1E35_014_102_024_C` — crop `000`

**Tokens (HIGH):** input=1240, output=141, thinking=1982

### OLD (HIGH thinking)
```
vingt-quatre Octobre à onze heure et demie du matin . Pardevant
nous Louis René Ruillier, Maire de la commune de l'Anse-
Bertrand, est comparu monsieur Ferdinand Ruillier, habitant-
propriétaire en cette commune lequel a déclaré que la négresse
Athalie, âgée de cinquante trois ans, appartenant à la société -
Ruillier Frères et fils, habitation Sainte Félicité, inscrite registre
E f.° 118 n° 3942, est décédée hier vingt trois du courant à trois heures
du soir. En foi de quoi nous avons dressé le présent acte .
```

### NEW (LOW thinking)
```
vingt-quatre Octobre à onze heure et demie du matin . Pardevant
nous Louis René Ruillier, Maire de la commune de l' Anse-
Bertrand, est comparu monsieur Ferdinand Ruillier, habitant
propriétaire en cette commune lequel a déclaré que la négresse
Athalie, âgée de cinquante trois ans, appartenant à la société
Ruillier Frères et fils, habitation Sainte Félicité, inscrite registre
E f.° 118 n.° 3942, est décédée hier vingt trois du courant à trois heures
du soir. En foi de quoi nous avons dressé le présent acte .
```

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Act type | Décès | Décès |
| Slave name | Athalie | Athalie |
| Sex / color | F / négresse | F / négresse |
| Age | 53 ans | 53 ans |
| Owner | société Ruillier Frères et fils | société Ruillier Frères et fils |
| Plantation | habitation Sainte Félicité | habitation Sainte Félicité |
| Parish | Anse-Bertrand | Anse-Bertrand |
| Registration | E f.° 118 n° 3942 | E f.° 118 n.° 3942 |
| Event date | 23 Octobre | 23 Octobre |
| Declaration date | 24 Octobre | 24 Octobre |
| Declarant | Ferdinand Ruillier | Ferdinand Ruillier |
| **Punctuation** | **"habitant-propriétaire", "société -"** | **"habitant propriétaire", "société"** |
| Note | Near-identical; minor hyphenation differences | |

---

## Plein Texte — `FRAD971_1E35_014_102_024_C` — crop `014`

**Tokens (HIGH):** input=1251, output=146, thinking=2015

### OLD (HIGH thinking)
```
L'An mil huit cent quarante deux, le vendredi quatorze
Octobre Pardevant nous Louis René Ruillier, Maire de la
commune de l'Anse-Bertrand, est comparu le Sieur
Fontan gérant l'habitation Laberthaudière sise en cette
commune, à madame veuve Godmart-Basmont, lequel
a déclaré que la négresse Angèle, âgée de vingt deux ans
inscrite régistre C folio 40 n° 1923, esclave de ladite habitation
est décédée hier au soir à six heures. En foi de quoi
nous avons dressé le présent acte...
```

### NEW (LOW thinking)
```
L'An mil huit cent quarante deux, le vendredi quatorze
Octobre Pardevant nous Louis René Ruillier, maire de la
commune de l'Anse-Bertrand, est comparu le Sieur
Fontan gérant l'habitation Laberthaudière sise en cette
commune, à madame veuve Godmart-Basmont, lequel
a déclaré que la négresse Angèle, âgée de vingt deux ans
inscrite régistre C folio 40 n° 1923, esclave de ladite habitation
est décédée hier au soir à six heures. En foi de quoi
nous avons dressé le présent acte...
```

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Act type | Décès | Décès |
| Slave name | Angèle | Angèle |
| Sex / color | F / négresse | F / négresse |
| Age | 22 ans | 22 ans |
| Owner | Mme veuve Godmart-Basmont | Mme veuve Godmart-Basmont |
| Plantation | habitation Laberthaudière | habitation Laberthaudière |
| Parish | Anse-Bertrand | Anse-Bertrand |
| Registration | C folio 40 n° 1923 | C folio 40 n° 1923 |
| Event date | hier au soir (13 Oct.) | hier au soir (13 Oct.) |
| Declaration date | 14 Octobre 1842 | 14 Octobre 1842 |
| Declarant | Sieur Fontan | Sieur Fontan |
| **Capitalization** | **"Maire"** | **"maire"** |
| Note | Identical content; only "Maire" vs "maire" differs | |

---

## Marge — `FRAD971_1E35_014_102_024_C` — crop `005`

**Tokens (HIGH):** input=1242, output=21, thinking=1972

### OLD (HIGH thinking)
```
N° 215
Décès de Gean (Saint-)
Hon Blonzac.
```

### NEW (LOW thinking)
```
N° 215
Décès de Gean (Saint-)
Hon,, Blonzac.
- ,, -
```

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Act type | Décès | Décès |
| Act number | 215 | 215 |
| Slave name | Gean (Saint-) | Gean (Saint-) |
| **Plantation** | **Hon Blonzac** | **Hon,, Blonzac** |
| **Extra text** | **(none)** | **"- ,, -"** |
| Note | Minor punctuation differences only | |

---

## Marge — `FRAD971_1E35_014_102_024_C` — crop `006`

**Tokens (HIGH):** input=1213, output=21, thinking=1863

### OLD (HIGH thinking)
```
N° 218
Décès de Virginie
Habi,,on Laberthandière
```

### NEW (LOW thinking)
```
N° 218
Décès de Virginie
Habi.on Laberthandière
```

### Key variables

| Variable | HIGH thinking (old) | LOW thinking (new) |
|----------|--------------------|--------------------|
| Act type | Décès | Décès |
| Act number | 218 | 218 |
| Slave name | Virginie | Virginie |
| **Plantation abbrev.** | **Habi,,on Laberthandière** | **Habi.on Laberthandière** |
| Note | Identical content; only abbreviation punctuation differs (,, vs .) | |

---


## Round 3 Cost Summary

```
Model: gemini-3.1-pro-preview
Total calls: 4
Input tokens:           4,946  (EUR 0.0084)
Output tokens:            329  (EUR 0.0033)
Thinking tokens:        7,832  (EUR 0.0797)
--------------------------------------------------
TOTAL ESTIMATED COST: EUR 0.0915
WARNING: 7,832 thinking tokens detected! Verify your Google billing matches this estimate.
```
