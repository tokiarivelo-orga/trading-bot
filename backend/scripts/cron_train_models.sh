#!/bin/bash
# Script to run bi-weekly training of all Deep Learning models
# Suitable for a cron job.

cd "$(dirname "$0")/../.." || exit 1

echo "================================================="
echo "Starting Bi-weekly Model Training"
echo "Date: $(date)"
echo "================================================="

echo ""
echo "--- Training XAUUSD (Model: smc_dl_m5) ---"
SYMBOL="XAUUSD" make train-dl

echo ""
echo "--- Training Step Index 200 (Model: smc_dl_m5_step200) ---"
SYMBOL="Step Index 200" make train-dl

echo ""
echo "================================================="
echo "Training finished at: $(date)"
echo "================================================="
