#!/bin/bash
# Run the full ESOL benchmark (Phase 1 + Phase 2) in parallel.
# Phase 1 is I/O-bound (HTTP to Ollama); Phase 2 uses ChemBERTa on GPU/CPU.
# They do not contend for resources.
#
# Usage:  bash run_benchmark.sh
# Logs:   logs/phase1.log   logs/phase2.log
# Stop:   kill $(cat .pids)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOGS="$REPO_ROOT/logs"
mkdir -p "$LOGS"

echo "============================================================"
echo "  Notation Matters — full ESOL benchmark"
echo "  $(date)"
echo "  Phase 1: 1072 molecules × 4 notations (Qwen3.6 via Ollama)"
echo "  Phase 2: augmented-SMILES mechanistic (ChemBERTa-2)"
echo "============================================================"

# Build notation cache first (fast, idempotent)
echo "[$(date +%T)] Building notation cache..."
python src/build_notation_cache.py

# Phase 2 — start first so ChemBERTa loads while Phase 1 is initialising
echo "[$(date +%T)] Launching Phase 2 (mechanistic)..."
nohup python src/mechanistic_analysis.py > "$LOGS/phase2.log" 2>&1 &
PID2=$!
echo "  PID: $PID2"

sleep 30  # let Phase 2 claim the GPU before Phase 1 starts

# Phase 1 — long-running, checkpoints every 25 molecules
echo "[$(date +%T)] Launching Phase 1 (notation benchmark)..."
nohup python src/notation_benchmark.py > "$LOGS/phase1.log" 2>&1 &
PID1=$!
echo "  PID: $PID1"

echo "$PID1 $PID2" > "$REPO_ROOT/.pids"

echo ""
echo "============================================================"
echo "  Both phases running in background."
echo "  Monitor:"
echo "    tail -f $LOGS/phase1.log"
echo "    tail -f $LOGS/phase2.log"
echo "  Results:"
echo "    results/phase1_full_results.json  (per-molecule predictions)"
echo "    results/phase1_full_stats.json    (summary statistics)"
echo "  Estimated time:"
echo "    Phase 2: 1–4 h   |   Phase 1: 5–8 h (safe to leave overnight)"
echo "============================================================"
