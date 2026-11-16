#!/usr/bin/env bash
set -u

# Train the layout detectors on data/layout_merged with the TrainingSettings /
# FasterRCNNSettings / RFDETRSettings budget, then score each best.pt on dev and
# test with parse.metrics. A model whose best.pt already exists is not retrained, so
# the script can be relaunched after an interruption: run without arguments once all
# models are trained, it only re-scores them. Pass run names to restrict it,
# e.g. `bash bench_layout.sh bench_frcnn`. EXTRA holds per-run train.py options.
# rfdetr is installed on the box only (not in pyproject.toml) when a bench_rfdetr*
# run is requested.
# DocLayout-YOLO runs under TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1: doclayout-yolo
# 0.0.4 reloads its own checkpoints with torch.load(f) (no weights_only), which
# torch >= 2.6 rejects; the variable only affects calls that leave weights_only unset.

cd ~/trocr_handwritten
source .venv/bin/activate
DATA=data/layout_merged
EVAL=$DATA/eval
mkdir -p "$EVAL" models

python -c "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" || {
    echo "CUDA not available to torch: see RUN_LAYOUT_BENCHMARK.md, section 2"; exit 1; }

DOCLAYOUT=models/doclayout_yolo_docsynth300k_imgsz1600.pt
[ -f "$DOCLAYOUT" ] || hf download juliozhao/DocLayout-YOLO-DocSynth300K-pretrain \
    doclayout_yolo_docsynth300k_imgsz1600.pt --local-dir models

declare -A BASE=(
    [bench_yolo11m_v2]=yolo11m.pt
    [bench_yolo26m]=yolo26m.pt
    [bench_doclayout]=$DOCLAYOUT
    [bench_rfdetr_m]=rfdetr_medium
    [bench_rtdetr_l]=rtdetr-l.pt
    [bench_frcnn]=fasterrcnn_resnet50_fpn
)
declare -A EXTRA=(
    [bench_frcnn]="--patience 10"
)
RUNS=${*:-bench_yolo11m_v2 bench_yolo26m bench_doclayout bench_rfdetr_m bench_rtdetr_l bench_frcnn}

if [[ " $RUNS " == *" bench_rfdetr"* ]] && ! python -c "import rfdetr" 2>/dev/null; then
    uv pip install "rfdetr[train]==1.12.1" || { echo "rfdetr install failed"; exit 1; }
fi

for NAME in $RUNS; do
    BEST=$DATA/models/$NAME/weights/best.pt
    PREFIX=""
    [ "$NAME" = bench_doclayout ] && PREFIX="env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
    if [ -f "$BEST" ]; then
        echo "skip training $NAME (found $BEST)"
    else
        echo "===== training $NAME (${BASE[$NAME]}) ====="
        $PREFIX python -m trocr_handwritten.parse.train --path-data "$DATA" \
            --images-dir "$DATA/images_src" --model-base "${BASE[$NAME]}" \
            --name "$NAME" ${EXTRA[$NAME]:-} 2>&1 | tee "$HOME/train-$NAME.log"
    fi
    for SPLIT in dev test; do
        echo "===== scoring $NAME on $SPLIT ====="
        $PREFIX python -m trocr_handwritten.parse.metrics --path-data "$DATA" \
            --split "$SPLIT" --model "$BEST" --device cuda \
            --out-json "$EVAL/${NAME}_${SPLIT}_cuda.json" 2>&1 | tee "$HOME/eval-$NAME-$SPLIT.log"
    done
done
ls -la "$EVAL"
echo "ALL DONE"
