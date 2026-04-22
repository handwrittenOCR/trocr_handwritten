# Déployer un modèle sur H100 Scaleway avec vLLM

Ce guide déploie un modèle Hugging Face sur une instance H100 Scaleway derrière une API OpenAI-compatible (vLLM), puis le branche au pipeline OCR via le provider `vllm`.

Un seul modèle est servi à la fois. Pour en changer, on tue le serveur et on relance le script avec un autre repo id.

## Modèles cibles

| Repo HF | Type | Gated |
|---|---|---|
| `google/gemma-4-26B-A4B-it` | Texte MoE | Oui (HF token requis) |
| `Qwen/Qwen3.6-35B-A3B` | Texte MoE | Non |
| `Teklia/Qwen2.5-VL-7B-DAI-CReTDHI-RecordGold-ATR` | Vision-Language | Non |

## 1. Provisionner l'instance

Dans la console Scaleway, créer une instance **H100-1-80G** (80 GB VRAM, Ubuntu 22.04). La clé SSH ajoutée au compte est propagée automatiquement.

Se connecter :

```bash
ssh root@<IP_PUBLIQUE>
```

Créer un workspace persistant :

```bash
mkdir -p /workspace && cd /workspace
```

## 2. Récupérer les scripts

Deux options.

**Option A — cloner le repo complet :**

```bash
git clone https://github.com/handwrittenOCR/trocr_handwritten.git
cd trocr_handwritten/trocr_handwritten/llm/deployment
```

**Option B — copier juste les 3 scripts depuis ta machine :**

```bash
# Depuis ton Mac
scp trocr_handwritten/llm/deployment/*.sh root@<IP_PUBLIQUE>:/workspace/
```

## 3. Installer l'environnement

```bash
bash setup_h100.sh
```

Ce script :
- installe les paquets système de base (`build-essential`, `tmux`, `nvtop`...)
- vérifie la présence du driver CUDA via `nvidia-smi`
- installe `uv`
- crée un venv Python 3.12 dans `/workspace/.venv`
- installe `vllm`, `huggingface_hub[cli]`, `hf_transfer`

Durée : ~3-5 minutes.

## 4. Télécharger les modèles

Gemma nécessite un token HF (modèle gated). Demander l'accès sur la page HF, puis :

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
bash download_models.sh
```

Les poids sont cachés dans `/workspace/hf_cache`. Les trois modèles pèsent environ 110-130 GB au total.

## 5. Servir un modèle

Le script prend le repo id en argument et un port optionnel (8000 par défaut) :

```bash
# Dans un tmux pour garder le serveur vivant après déconnexion SSH
tmux new -s vllm

bash serve_vllm.sh Qwen/Qwen3.6-35B-A3B
# ou
bash serve_vllm.sh google/gemma-4-26B-A4B-it
# ou (vision)
bash serve_vllm.sh Teklia/Qwen2.5-VL-7B-DAI-CReTDHI-RecordGold-ATR
```

Détacher tmux avec `Ctrl+b` puis `d`. Revenir avec `tmux attach -t vllm`.

Le serveur écoute sur `http://0.0.0.0:8000/v1`. Le nom exposé côté API (`--served-model-name`) est la dernière portion du repo, ex : `Qwen3.6-35B-A3B`.

## 6. Exposer le port côté client

Deux options.

**Tunnel SSH (recommandé, pas besoin d'ouvrir de port) :**

```bash
# Depuis ton Mac, dans un autre terminal
ssh -N -L 8000:localhost:8000 root@<IP_PUBLIQUE>
```

Le serveur est alors joignable sur `http://localhost:8000/v1` depuis ton Mac.

**IP publique (plus simple mais à sécuriser) :**

Ouvrir le port 8000 dans le security group Scaleway. L'endpoint devient `http://<IP_PUBLIQUE>:8000/v1`. Ajouter `--api-key <token>` à `vllm serve` et passer le token côté client via `VLLM_API_KEY` pour éviter un endpoint ouvert.

## 7. Configurer le client OCR

Dans le `.env` du projet local :

```bash
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY     # ou le token si --api-key utilisé côté serveur
```

Vérifier que le serveur répond :

```bash
curl $VLLM_BASE_URL/models
```

## 8. Lancer l'OCR

Le paramètre `--model` est le nom exposé par vLLM (`--served-model-name`), pas forcément le repo id complet.

```bash
uv run python -m trocr_handwritten.llm.ocr \
    --provider vllm \
    --model Qwen3.6-35B-A3B \
    --input_dir data/processed/images \
    --pattern "*/*/*.jpg" \
    --output_dir data/ocr/test/predictions/qwen3-35b \
    --max_concurrent 16
```

Surcharger l'URL directement en ligne de commande (utile pour pointer vers un autre host sans toucher au `.env`) :

```bash
uv run python -m trocr_handwritten.llm.ocr \
    --provider vllm \
    --model Gemma-4-26B-A4B-it \
    --vllm_base_url http://localhost:8000/v1 \
    --input_dir data/processed/images \
    --output_dir data/ocr/test/predictions/gemma-4-26b
```

## 9. Changer de modèle

```bash
# Dans le tmux vllm
Ctrl+C    # tue le serveur
bash serve_vllm.sh <autre_modele>
```

Pas besoin de re-télécharger : les poids restent dans `/workspace/hf_cache`.

## Dépannage

**`No space left on device` pendant `download_models.sh`** — le disque root `/` des instances Scaleway H100 fait ~110 GB, trop peu pour les 3 modèles (~130 GB). Un volume NVMe de ~2.7 TB est monté sur `/scratch`. Basculer le cache HF :

```bash
# Nettoyer les downloads partiels sur /
rm -rf /workspace/hf_cache/models--*<partiel>*

# Déplacer ce qui a déjà été téléchargé
mkdir -p /scratch/hf_cache
mv /workspace/hf_cache/* /workspace/hf_cache/.[!.]* /scratch/hf_cache/ 2>/dev/null || true

# Remplacer le dossier par un symlink
rm -rf /workspace/hf_cache
ln -s /scratch/hf_cache /workspace/hf_cache

# Vérifier puis relancer
df -h / /scratch
bash download_models.sh
```

Vérifier à l'avance avec `lsblk` : si un `/dev/sdb` ou `/dev/nvme1n1` de plusieurs TB est listé et monté sur `/scratch`, utiliser ce chemin. Si rien n'est monté, le formater et le monter manuellement (`mkfs.ext4` + `mount`).

**`CUDA out of memory` au chargement** — baisser `--gpu-memory-utilization` (dans `serve_vllm.sh`) à `0.85` ou `--max-model-len` à `16384`.

**`Model not found` côté client** — vérifier le nom exact avec `curl $VLLM_BASE_URL/models`. vLLM expose la dernière portion du repo id sauf si `--served-model-name` est override.

**Téléchargement lent** — vérifier que `hf_transfer` est bien installé (`HF_HUB_ENABLE_HF_TRANSFER=1` dans `download_models.sh`). Sur Scaleway la bande passante est généralement > 500 MB/s.

**MoE qui refuse de charger** — ajouter `--enforce-eager` dans `serve_vllm.sh` pour désactiver CUDA graphs si le compile échoue.

**Qwen2.5-VL ne reçoit pas l'image** — le provider envoie déjà `image_url` en base64, géré nativement par vLLM. Si l'image n'est pas prise en compte, vérifier que `--limit-mm-per-prompt image=4` est bien passé (c'est fait automatiquement par `serve_vllm.sh` pour tout repo contenant `VL`).

## Coûts

vLLM étant self-hosted, le coût ne dépend pas des tokens mais du temps d'occupation de l'instance H100. Le `CostTracker` calcule :

```
cost_usd = elapsed_hours × VLLM_HOURLY_EUR × EUR_TO_USD
```

avec `VLLM_HOURLY_EUR=2.8` et `EUR_TO_USD=1.08` par défaut dans [cost_tracker.py](../utils/cost_tracker.py). Les tokens restent comptés et exposés dans le `summary.json`, mais ne sont pas facturés.

Si tu changes de plan Scaleway, ajuste `VLLM_HOURLY_EUR`. Pour ajouter un nouveau modèle self-hosté, ajoute son nom exposé (celui passé à `--served-model-name`) dans le set `VLLM_MODELS`.
