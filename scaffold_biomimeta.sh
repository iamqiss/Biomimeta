#!/usr/bin/env bash
set -euo pipefail

echo "Scaffolding Biomimeta research suite..."

# Top-level dirs
mkdir -p studies/bav/{stimuli,protocols,results,figures,reports}
mkdir -p studies/cpa/{datasets,ladders,scripts,results,figures,reports}
mkdir -p studies/pqs/{stimuli_manifests,protocols,irb,results,figures,reports}
mkdir -p tools/{metrics,sims,plots}
mkdir -p ci/github/workflows

# --------- BAV README -------------
cat > studies/bav/README.md <<'MD'
# Biological Accuracy Validation (BAV)

This suite checks whether Biomimeta’s visual pathway modules reproduce
known biological response patterns.

- **Stimuli:** gratings, natural movies, saccade profiles
- **Metrics:** correlation vs reference CSF, tuning RMSE, KL-divergence
- **Reports:** auto-generated PDF in `/studies/bav/reports`

Run with (placeholder CLI):
```bash
# Example (implement CLI/binary later)
cargo run --release -- --suite bav --stimuli studies/bav/stimuli/manifest.yaml
