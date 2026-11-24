# Runbook — baselines VLM zero-shot sur H100 (test v2.0)

Corpus : `data/ocr/20260807-ft-data` (test = 652 images).
Déjà fait : `zs-qwen3-vl-8b`, `zs-qwen3-vl-2b`.
Restent **6 modèles** (voir table §3).

Approche par défaut : **serveur vLLM sur la H100, inférence depuis le Mac via tunnel SSH**
→ les `.md` s'écrivent en local sur le Mac (pas de rsync retour), à côté des 2 déjà faits.

---

## 1. Rallumer le GPU (CLI)

```bash
export SCW_ACCESS_KEY="$SCW_ACCESS_KEY" SCW_SECRET_KEY="$SCW_SECRET_KEY"
export SCW_DEFAULT_PROJECT_ID="$SCW_PROJECT_ID" SCW_DEFAULT_ZONE=fr-par-2

scw instance server create \
  type=H100-1-80G zone=fr-par-2 \
  image=ubuntu_jammy_gpu_os_12 root-volume=block:500GB \
  name=htr-h100 ip=new --wait

# récupérer l'IP (elle change à chaque instance)
IP=$(scw instance server list zone=fr-par-2 name=htr-h100 -o json | jq -r '.[0].public_ip.address')
echo "$IP"
```

Provisionner la box (l'image n'a pas `/workspace`) :
```bash
ssh root@$IP
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /workspace
export HF_TOKEN=hf_xxx          # requis pour Gemma (gated)
# récupérer le code (repo + .sh + tes modifs) : voir §2 pour le rsync depuis le Mac
cd /root/trocr_handwritten
bash trocr_handwritten/llm/deployment/setup_h100.sh
bash trocr_handwritten/llm/deployment/download_models.sh
```

---

## 2. Données / code sur l'instance

**Via tunnel (défaut) : les images ne sont PAS nécessaires sur la box** — `ocr.py` tourne
sur le Mac et lit les images locales ; seul le serveur a besoin des poids (téléchargés par vLLM).
Il faut juste le **code** à jour sur la box (tes modifs ne sont pas poussées sur git) :

```bash
# depuis le Mac, racine du repo
rsync -avz --progress --exclude .venv --exclude .git --exclude data \
  --exclude models --exclude __pycache__ --exclude '*.pyc' \
  ./ root@$IP:/root/trocr_handwritten/
```

**Si tu préfères tout faire sur la box** (option sans tunnel), il faut aussi le split test :
```bash
# vérifier côté box
ssh root@$IP 'find /root/trocr_handwritten/data/ocr/20260807-ft-data/test/images -type f | wc -l'  # attendu 652
# sinon, le monter depuis le Mac
ssh root@$IP 'mkdir -p /root/trocr_handwritten/data/ocr/20260807-ft-data'
rsync -avz --progress data/ocr/20260807-ft-data/test \
  root@$IP:/root/trocr_handwritten/data/ocr/20260807-ft-data/
```

---

## 3. Lancer les 6 modèles manquants

Ouvre le **tunnel** (Mac, à laisser tourner) :
```bash
ssh -N -L 8000:localhost:8000 root@$IP
```

Pour chaque modèle : **servir sur la box** (fenêtre A) puis **inférer depuis le Mac** (fenêtre B),
Ctrl‑C sur A quand fini, modèle suivant.

### Fenêtre A — serveur (sur la box)
`--mm-processor-cache-gb 0` évite le bug 500 du cache multimodal ; `--served-model-name` = alias basename.
```bash
source /workspace/.venv/bin/activate
REPO=Qwen/Qwen3-VL-32B-Instruct           # <-- change selon la table
ALIAS=${REPO##*/}
vllm serve "$REPO" --host 0.0.0.0 --port 8000 \
  --served-model-name "$ALIAS" \
  --dtype bfloat16 --max-model-len 81292 --gpu-memory-utilization 0.95 \
  --trust-remote-code --limit-mm-per-prompt '{"image":1}' \
  --download-dir /workspace/hf_cache \
  --mm-processor-cache-gb 8
```
(les 32B/31B/35B : `serve_vllm.sh` baisse ctx à 16384 / util 0.95 ; si OOM, ajoute `--max-model-len 16384 --gpu-memory-utilization 0.95`.)

### Fenêtre B — inférence (sur le Mac)
```bash
cd /Users/agomberto/Desktop/trocr_handwritten
ROOT=data/ocr/20260807-ft-data
ALIAS=Qwen3-VL-32B-Instruct                # <-- même alias que ci-dessus
OUT=zs-qwen3-vl-32b                         # <-- dossier selon la table
uv run python -m trocr_handwritten.llm.ocr --provider vllm \
  --model "$ALIAS" --vllm_base_url http://localhost:8000/v1 \
  --input_dir "$ROOT/test/images" --pattern "*/*.jpg" \
  --output_dir "$ROOT/test/predictions/$OUT" \
  --repeat 5 --temperature 0 --max_tokens 1024 --max_concurrent 32 --disable_thinking
```

Gros modèles, un à la fois, TP=2 :
vllm serve Qwen/Qwen3-VL-32B-Instruct --port 8000 \
  --tensor-parallel-size 2 \
  --served-model-name Qwen3-VL-32B-Instruct --dtype bfloat16 \
  --max-model-len 16384 --gpu-memory-utilization 0.92 --trust-remote-code \
  --limit-mm-per-prompt '{"image":4}' --download-dir /workspace/hf_cache --mm-processor-cache-gb 0
(garde --max-model-len 16384 pour les 32B/31B/35B ; c'est large vu que la sortie max ≈ 490 tokens.)


### Table des 6 restants (repo → alias → dossier → thinking)
| `REPO` (servir) | `ALIAS` = `--model` | `$OUT` | flag |
|---|---|---|---|
| `Qwen/Qwen3-VL-32B-Instruct` | `Qwen3-VL-32B-Instruct` | `zs-qwen3-vl-32b` | `--disable_thinking` |
| `Qwen/Qwen3.6-35B-A3B` | `Qwen3.6-35B-A3B` | `zs-qwen3.6-35b` | `--disable_thinking` |
| `google/gemma-4-12B-it` | `gemma-4-12B-it` | `zs-gemma-4-12b` | — |
| `google/gemma-4-26B-A4B-it` | `gemma-4-26B-A4B-it` | `zs-gemma-4-26b` | — |
| `google/gemma-4-31B-it` | `gemma-4-31B-it` | `zs-gemma-4-31b` | — |
| `Teklia/Qwen2.5-VL-7B-DAI-CReTDHI-RecordGold-ATR` | `Qwen2.5-VL-7B-DAI-CReTDHI-RecordGold-ATR` | `zs-qwen2.5-vl-teklia` | — |

Retire `--disable_thinking` pour Gemma et Teklia. Rappels : `--max_tokens 2048` (max réel ≈ 490 tokens),
`--temperature 0.3` (sinon 5 runs identiques). Vérifie l'alias servi : `curl -s localhost:8000/v1/models`.

---

## 4. Scoring (une fois les 8 modèles produits, sur le Mac)
```bash
ROOT=data/ocr/20260807-ft-data; P=$ROOT/test/predictions; L=$ROOT/test/labels
for M in zs-qwen3-vl-2b zs-qwen3-vl-8b zs-qwen3-vl-32b zs-qwen3.6-35b \
         zs-gemma-4-12b zs-gemma-4-26b zs-gemma-4-31b zs-qwen2.5-vl-teklia; do
  uv run python -m trocr_handwritten.llm.metrics --predictions "$P/$M"/run{1..5} \
    --labels "$L" --output "$P/$M/metrics.json"
  uv run python -m trocr_handwritten.llm.stats ci --metrics "$P/$M/metrics.json" --level N2
done
```

---

## 5. Éteindre pour ne rien payer
Poweroff ≠ gratuit sur GPU Scaleway → **supprimer** serveur + volume + IP :
```bash
SID=$(scw instance server list zone=fr-par-2 name=htr-h100 -o json | jq -r '.[0].id')
scw instance server terminate "$SID" zone=fr-par-2 with-ip=true with-block=true
scw instance server list zone=fr-par-2   # doit être vide
scw instance volume list  zone=fr-par-2
scw instance ip list      zone=fr-par-2
```
Via tunnel, les `.md` sont déjà sur le Mac → rien à sauver sur la box avant le terminate.
