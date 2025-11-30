#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py -m experiment=deepcycle_10 \
  model=deepcycle  \
  launcher=ox_a10 \
  datamodule=deepcycle \
  seed=42,43,44 &