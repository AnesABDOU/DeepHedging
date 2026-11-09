foreach ($c in 0, 0.0005, 0.001, 0.005, 0.01) {
  foreach ($s in 0, 1, 2) {
    uv run python scripts/train.py --config configs/tc.yaml --c $c --seed $s
  }
}