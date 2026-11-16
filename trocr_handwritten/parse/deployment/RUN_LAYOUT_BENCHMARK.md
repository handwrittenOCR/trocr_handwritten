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

Données, deux possibilités :
- **depuis le Mac de l'auteur**, en flux tar (plus rapide que rsync pour beaucoup de petits fichiers ; les liens relatifs de `images_src/` restent valides car les trois dossiers partent ensemble) :
  ```bash
  COPYFILE_DISABLE=1 tar --no-xattrs -cf - --exclude 'data/layout_merged/models' --exclude '*.html' --exclude 'labels.cache' \
    data/raw/initial data/layout_hf/images data/layout_merged \
    | ssh root@$IP 'tar -xf - -C /root/trocr_handwritten && echo TAR_OK'
  ```
- **depuis Hugging Face**, sur la box (même dataset, format reconstruit) : `python -m trocr_handwritten.parse.hf_layout --out-dir data/layout_merged`, puis utiliser `--images-dir data/layout_merged/images_src` (valeur par défaut du benchmark).

---

## 3. Lancer le benchmark (sur la box)

Script : `python -m trocr_handwritten.parse.benchmark [runs…]` (réglages dans `BenchmarkSettings`, `parse/settings.py`). Pour chaque run : entraînement (`parse.train`), puis scoring dev et test (`parse.metrics`) en JSON dans `data/layout_merged/eval/`, logs dans `data/layout_merged/logs/`. Un run n'est sauté que s'il a son marqueur `models/<run>/TRAINING_DONE`, écrit en fin d'entraînement réussi : un run interrompu (qui peut déjà avoir un `best.pt`) est réentraîné. Les poids DocLayout sont téléchargés au premier besoin ; `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` est posé pour ce seul run.

Deux sessions tmux en parallèle (≈ 38 Go de GPU au pic sur la L40S). `remain-on-exit` garde le panneau ouvert si un processus meurt, pour lire l'erreur.
```bash
cd ~/trocr_handwritten && source .venv/bin/activate
tmux set -g remain-on-exit on
tmux new -d -s benchA 'python -m trocr_handwritten.parse.benchmark bench_yolo11m_v2 bench_doclayout bench_rtdetr_l 2>&1 | tee ~/benchA.log; bash'
sleep 60
tmux new -d -s benchB 'python -m trocr_handwritten.parse.benchmark bench_yolo26m bench_rfdetr_m bench_frcnn 2>&1 | tee ~/benchB.log; bash'
tmux ls
```
Suivre en direct depuis le Mac (sans s'attacher ; si on s'attache, détacher par **Ctrl-b puis d**) :
```bash
ssh -t root@$IP "watch -n 30 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; for f in ~/trocr_handwritten/data/layout_merged/logs/train-*.log; do echo \"== \$f\"; tail -c 300 \$f | tr \"\\r\" \"\\n\" | tail -n 2; done'"
```
Si une session tombe, relancer la même commande : les runs finis sont sautés.

Fin attendue : `ALL DONE` dans les deux sessions, un `TRAINING_DONE` par run, 12 JSON dans `data/layout_merged/eval/`.

Réévaluation sur GPU libre (les temps par page mesurés pendant le parallèle sont faussés) :
```bash
python -m trocr_handwritten.parse.benchmark --score-only
```

---

## 4. Rapatrier les résultats (Mac)

```bash
mkdir -p data/layout_merged/eval data/layout_merged/logs
rsync -avz root@$IP:/root/trocr_handwritten/data/layout_merged/models/ data/layout_merged/models/ \
  --include 'bench_*/***' --exclude '*'
rsync -avz root@$IP:/root/trocr_handwritten/data/layout_merged/eval/ data/layout_merged/eval/
rsync -avz root@$IP:/root/trocr_handwritten/data/layout_merged/logs/ data/layout_merged/logs/
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
python -m trocr_handwritten.parse.benchmark --score-only --device cpu
```
Écrit `data/layout_merged/eval/<run>_{dev,test}_cpu.json` pour chaque run qui a un `best.pt`. Chaque JSON contient aussi `conf_sweep` : P/R/F1 pour des seuils de confiance de 0,1 à 0,9, pour choisir un seuil sur le dev.
