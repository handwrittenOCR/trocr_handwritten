# Runbook — benchmark des 6 détecteurs de régions sur L40S (Scaleway)

Modèles (run) : YOLO11m (`bench_yolo11m_v2`), YOLO26-M (`bench_yolo26m`), DocLayout-YOLO
DocSynth300K (`bench_doclayout`), RF-DETR-M (`bench_rfdetr_m`), RT-DETR-l (`bench_rtdetr_l`),
Faster R-CNN R50-FPN (`bench_frcnn`).
Budget commun (`TrainingSettings` / `FasterRCNNSettings` / `RFDETRSettings`) : 100 époques max,
patience 20 (Faster R-CNN : 10), 1024 px, batch 8, `best.pt` = meilleur mAP50:95 dev.
Données : `data/layout_merged/` (labels corrigés selon les conventions du 13/11).
Deux sessions tmux en parallèle sur la même L40S, ~3 à 4 h au total (à recaler après les premières époques).

Toutes les commandes se lancent **depuis la racine du repo sur le Mac**, sauf indication « sur la box ».

---

## 1. Créer l'instance

```bash
export SCW_DEFAULT_ZONE=fr-par-2
scw instance server create \
  type=L40S-1-48G zone=fr-par-2 \
  image=ubuntu_jammy_gpu_os_12 root-volume=block:100GB \
  name=layout-l40s ip=new --wait

IP=$(scw instance server list zone=fr-par-2 name=layout-l40s -o json | jq -r '.[0].public_ip.address')
echo "$IP"
```

---

## 2. Code, données et environnement

Code (sans `.env*`, venvs, `.git`, données ni poids) :
```bash
rsync -avz --exclude '.env*' --exclude '.venv*' --exclude .git --exclude .claude \
  --exclude data --exclude models --exclude adapters --exclude runs \
  --exclude __pycache__ --exclude '*.pyc' --exclude '*.pt' \
  ./ root@$IP:/root/trocr_handwritten/
```

Données (les liens relatifs de `images_src/` restent valides car les trois dossiers partent ensemble) :
```bash
rsync -avzR --exclude 'models/' --exclude '*.html' --exclude 'labels.cache' \
  data/raw/initial data/layout_hf/images data/layout_merged \
  root@$IP:/root/trocr_handwritten/
```

Sur la box :
```bash
ssh root@$IP
cd ~/trocr_handwritten
find data/layout_merged/images_src -xtype l | wc -l        # attendu 0 (aucun lien cassé)
apt-get install -y tmux
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
uv sync --python 3.11
source .venv/bin/activate
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # attendu True
```

Si la dernière ligne affiche `False` (driver trop ancien pour le build cu130 de PyPI) :
```bash
uv pip install --reinstall torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # attendu 2.13.0+cu126 True
```

---

## 3. Lancer le benchmark (sur la box)

Session A d'abord (elle télécharge DocLayout), puis B quand A affiche `===== training bench_yolo11m_v2`.
B installe `rfdetr[train]==1.12.1` dans le venv avant RF-DETR.
```bash
cd ~/trocr_handwritten
tmux new -d -s benchA 'bash trocr_handwritten/parse/deployment/bench_layout.sh bench_yolo11m_v2 bench_doclayout bench_rtdetr_l; bash'
tmux new -d -s benchB 'bash trocr_handwritten/parse/deployment/bench_layout.sh bench_yolo26m bench_rfdetr_m bench_frcnn; bash'
```
Suivre en direct depuis le Mac :
```bash
ssh -t root@$IP tmux attach -t benchA        # Ctrl-b d pour détacher ; idem benchB
ssh -t root@$IP "watch -n 30 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; for f in ~/train-*.log; do echo \"== \$f\"; tail -c 300 \$f | tr \"\\r\" \"\\n\" | tail -n 2; done'"
```
Si une session tombe (OOM compris), relancer la même commande : les modèles déjà entraînés sont sautés.
Si RF-DETR sature la mémoire en parallèle : le relancer seul quand A a fini.

Fin attendue : `ALL DONE` dans les deux sessions, 6 dossiers `data/layout_merged/models/bench_*`.

Réévaluation sur GPU libre (les temps par page mesurés pendant le parallèle sont faussés) :
```bash
tmux new -s rescore 'bash trocr_handwritten/parse/deployment/bench_layout.sh; bash'
```
Fin attendue : `skip training` pour les 6 runs, `ALL DONE`, 12 JSON dans `data/layout_merged/eval/`.

---

## 4. Rapatrier les résultats (Mac)

```bash
mkdir -p data/layout_merged/eval data/layout_merged/logs
rsync -avz root@$IP:/root/trocr_handwritten/data/layout_merged/models/ data/layout_merged/models/ \
  --include 'bench_*/***' --exclude '*'
rsync -avz root@$IP:/root/trocr_handwritten/data/layout_merged/eval/ data/layout_merged/eval/
rsync -avz root@$IP:'/root/*bench_*.log' data/layout_merged/logs/
ls data/layout_merged/models/bench_*/weights/best.pt | wc -l    # attendu 7 (6 + ancien bench_yolo11m)
ls data/layout_merged/eval/*_cuda.json | wc -l                  # attendu 12
```

---

## 5. Supprimer tout ce qui est facturé (Mac)

Un serveur éteint reste facturé : il faut **supprimer** serveur, volume et IP.
```bash
SID=$(scw instance server list zone=fr-par-2 name=layout-l40s -o json | jq -r '.[0].id')
scw instance server terminate "$SID" zone=fr-par-2 with-ip=true with-block=true
```
Vérifications (chaque liste doit être vide de toute ressource `layout-l40s`) :
```bash
scw instance server list zone=fr-par-2
scw instance volume list zone=fr-par-2
scw block volume list zone=fr-par-2
scw block snapshot list zone=fr-par-2
scw instance ip list zone=fr-par-2
```
Si un volume `block` subsiste : `scw block volume delete <id> zone=fr-par-2`.

---

## 6. Temps par page sur CPU (Mac, gratuit)

```bash
for R in bench_yolo11m_v2 bench_yolo26m bench_doclayout bench_rfdetr_m bench_rtdetr_l bench_frcnn; do
  P=""; [ "$R" = bench_doclayout ] && P="env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
  $P python -m trocr_handwritten.parse.metrics --path-data data/layout_merged --split test \
    --model data/layout_merged/models/$R/weights/best.pt --device cpu \
    --out-json data/layout_merged/eval/${R}_test_cpu.json
done
```
