#!/bin/bash
# Full ESOL benchmark launcher
# Phase 1 (LLM I/O) and Phase 2 (ChemBERTa CPU) run in PARALLEL:
#   - Phase 1 is I/O-bound (HTTP waits) → minimal CPU
#   - Phase 2 uses ChemBERTa on GPU/CPU → no conflict with Ollama
#
# Usage: bash run_full_benchmark.sh
# Logs:  logs/phase1_full.log  and  logs/phase2_full.log
# Stop:  cat /home/snu/workspace/notation_research/pids_full.txt
#        kill <pid1> <pid2>

set -euo pipefail

WORK="/home/snu/workspace/notation_research"
LOGS="$WORK/logs"
mkdir -p "$LOGS"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "============================================================"
echo "  FULL ESOL BENCHMARK — $(date)"
echo "  Phase 1: 1128 molecules × 4 notations (LLM)"
echo "  Phase 2: augmented-SMILES mechanistic (ChemBERTa)"
echo "============================================================"

cd "$WORK"

# ── Phase 2 first (fast, ~30-60 min on GPU, ~2-4h on CPU) ────────────────────
echo "[$(date +%T)] Launching Phase 2 (mechanistic)..."
nohup python3 08_full_esol_phase2.py \
    > "$LOGS/phase2_full.log" 2>&1 &
PID2=$!
echo "  Phase 2 PID: $PID2"

# Small stagger so Phase 2 loads model first (avoids GPU contention at start)
sleep 30

# ── Phase 1 (long, ~5-8 hours) ───────────────────────────────────────────────
echo "[$(date +%T)] Launching Phase 1 (4-notation LLM benchmark)..."
nohup python3 07_full_esol_phase1.py \
    > "$LOGS/phase1_full.log" 2>&1 &
PID1=$!
echo "  Phase 1 PID: $PID1"

# Save PIDs for later
echo "$PID1 $PID2" > "$WORK/pids_full.txt"
echo "PIDs saved to pids_full.txt"

echo ""
echo "============================================================"
echo "  Both experiments running in background."
echo "  Monitor with:"
echo "    tail -f $LOGS/phase1_full.log"
echo "    tail -f $LOGS/phase2_full.log"
echo ""
echo "  Results will appear in:"
echo "    $WORK/phase1_full_results.json"
echo "    $WORK/phase1_full_stats.json"
echo "    $WORK/phase2_full_results.json"
echo "    $WORK/phase2_full_stats.json"
echo ""
echo "  Estimated completion:"
echo "    Phase 2: ~1-4 hours"
echo "    Phase 1: ~5-8 hours  (safe to leave overnight)"
echo "============================================================"
