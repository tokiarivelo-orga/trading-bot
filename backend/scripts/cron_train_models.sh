#!/bin/bash
# Script to run bi-weekly training of all Deep Learning models
# Suitable for a cron job.
#
# NOTE ON THE Aug 9 2026 INCIDENT: a retrain shifted the model's output
# distribution and the bot went silent for days, because at the time
# smc_dl_m5_step200_v1.py gated entries directly on the model's directional
# softmax (min_direction_conf/min_tp_prob). That strategy has since been
# rewritten (see its module docstring) to an OB-first design where zone
# structure proposes the trade and the model only vetoes it when confidently
# opposed (`model_veto_threshold`) — there is no longer a hard confidence
# gate for a retrain to silently drift past, so no post-retrain threshold
# patch is needed here. `scripts/optimize_step200.py` still sweeps the old
# gate's thresholds and is kept for reference only; it does not describe the
# strategy that is actually live.

cd "$(dirname "$0")/.." || exit 1

PYTHON="${PYTHON:-$(which python3)}"
# Prefer the project virtualenv if it exists
if [ -f ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
fi

echo "================================================="
echo "Starting Bi-weekly Model Training"
echo "Date: $(date)"
echo "Python: $PYTHON"
echo "================================================="

echo ""
echo "--- Training XAUUSD (Model: smc_dl_m5) ---"
SYMBOL="XAUUSD" $PYTHON scripts/train_smc_dl.py
echo "XAUUSD training exit code: $?"

echo ""
echo "--- Training Step Index 200 (Model: smc_dl_m5_step200) ---"
SYMBOL="Step Index 200" $PYTHON scripts/train_smc_dl.py
TRAIN_RC=$?
echo "Step Index 200 training exit code: $TRAIN_RC"

echo ""
echo "================================================="
echo "Training + calibration finished at: $(date)"
echo "================================================="
