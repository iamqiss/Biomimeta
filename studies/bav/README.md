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
