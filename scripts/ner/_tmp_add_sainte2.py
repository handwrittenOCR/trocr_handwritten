import json

p = "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/3. OCR/2. TrOCR/5. Data (output)/ECES/NER_datasets/llm/owner_pairs/sainte_clusters.json"
with open(p, encoding="utf-8") as f:
    d = json.load(f)
by_canon = {c["canonical"]: c for c in d["owner_clusters"]}
by_plant = {c["canonical"]: c for c in d["plantation_clusters"]}


def E(canon, vs):
    if canon in by_canon:
        for v in vs:
            if v not in by_canon[canon]["variants"]:
                by_canon[canon]["variants"].append(v)


def P(canon, vs):
    if canon in by_plant:
        for v in vs:
            if v not in by_plant[canon]["variants"]:
                by_plant[canon]["variants"].append(v)


E("Monsieur Clavié", ["Clavié"])
E("héritiers Delaire", ["Delair"])
E(
    "Louis Cornette de Venancourt",
    [
        "M^er de Venancourt",
        "M^r de Venancourt",
        "Mier de Venancourt",
        "Monsieur de Venancourt",
        "Mr de Senoncourt",
    ],
)
E("Dame Veuve de Waresquiel", ["Mr de Waroquier", "Dame veuve de Wawrechin"])
E("héritiers Mollenthiel (Durivage)", ["Mollenthiel héritiers"])
E("Mr ffrench", ["Monsieur Ffrench", "P. french"])
E("Mr Hippolite", ["Sieur Hyppolite"])
E("Mr d'Ourville", ["Monsieur d'Ouville"])
E("Mr Corneille", ["le Sieur Corneille"])
E("Mr Emeran", ["le Sieur Emeran"])
E("Lamothe aîné", ["Mr Lamothe"])
E("Mr Martinet", ["Martinot"])
E("Mr Ledeuff", ["Ledneff"])
E("Dame Veuve Burat", ["mme ve Buriat"])
E("Dame veuve Dieupart Buel", ["Dieupart Ruelle"])
E("Olivier Bonnet (Gissac)", ["Monsieur Olivier Bonnet", "Mr Olivier Bonnet"])
E("Mr Petit le Brun", ["Letot le Brun"])
E(
    "Dame Veuve Budan (Dupaty)",
    [
        "Mad, Ve, Buddn",
        "Mad. Ve Budan",
        "Ve. Budan",
        "dame Ve Budan",
        "dame veuve Audan",
        "veuve Budan",
        "Dame Veuve Burdan",
        "Dame Veuve Courat",
    ],
)
E(
    "Couppé de Lahougrais",
    ["Dame Veuve Couppé de la Rougerie", "Dame Veuve Pouppé de la hongrais"],
)

new_widow_clusters = [
    {
        "canonical": "Dame Veuve Bourgoin",
        "family": "Bourgoin",
        "variants": ["Dame Veuve Bourgoin"],
    },
    {
        "canonical": "Dame Veuve Petit Le Brun (sainte)",
        "family": "Petit Le Brun",
        "variants": ["Dame Veuve Petit Le Brun"],
    },
    {
        "canonical": "Dame veuve Emeran (sainte)",
        "family": "Emeran",
        "variants": ["Dame veuve Emeran"],
    },
    {
        "canonical": "Dame Veuve Beaubrun",
        "family": "Beaubrun",
        "variants": ["Dame Veuve Beaubrun"],
    },
]
for nc in new_widow_clusters:
    if nc["canonical"] not in by_canon:
        d["owner_clusters"].append(nc)
        by_canon[nc["canonical"]] = nc

P("Bel-Etang", ["Bel-étang", "Belétang"])
P("Bellecour", ["Belcour", "Bellecour (Bérard)"])
P(
    "Néron Surgy et autres",
    [
        "Néron - Surgy et autres",
        "Néron Surgis et autres",
        "Néron surgy et autres",
        "Néron, Surgy et autres",
    ],
)
P("Riche Plaine", ["Riche-Plaine", "Richeplaine"])
P("Valeras", ["Valera", "Valerats", "Valéras"])
P("Cinq-Etangs", ["cinq-Etangs"])
P("d'Ourville", ["D'Ouville", "d'aurille", "d'ourille", "d'ourville"])
P("Hélou", ["hélou"])
P("Mahaudière", ["La Mahaudière", "la Mahautière", "Mahaudière (héritiers Pédèmon)"])
P("Sainte Rose", ["sainte-Rose"])
P("Sainte Marguerite", ["Ste Marguerite"])
P("Marly", ["Mably"])
P("Lary", ["Léry"])
P("Gissac", ["Pissac"])
P("Dupaty", ["Duhaty", "Dupraty"])
P("Delaire", ["Delair"])
P(
    "Papin et co-associés",
    [
        "Chapin et Co-associés",
        "Delpin et co-associés",
        "Sapin et associés",
        "Sapin, et associés",
        "Tapin et co-associés",
        "Tapion et co-associés",
        "dapien et co-associés",
        "lepin et co-associés",
        "sapin et s'associés",
        "tapin et co-associés",
    ],
)
P("Bourg de Sainte Anne", ["maison au Bourg de Sainte-Anne"])

# Remove duplicate Néron Surgy plantation if it exists alongside Veuve Néron Surgy
to_remove = []
for c in d["plantation_clusters"]:
    if c.get("canonical") == "Néron Surgy":
        to_remove.append(c)
        # Move its variants to Néron Surgy et autres
        for v in c["variants"]:
            if v not in by_plant["Néron Surgy et autres"]["variants"]:
                by_plant["Néron Surgy et autres"]["variants"].append(v)
for c in to_remove:
    d["plantation_clusters"].remove(c)

d["owner_clusters"] = [c for c in d["owner_clusters"] if c.get("variants")]
d["plantation_clusters"] = [c for c in d["plantation_clusters"] if c.get("variants")]
with open(p, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=2)
print(
    f"Owners: {len(d['owner_clusters'])}, Plantations: {len(d['plantation_clusters'])}"
)
