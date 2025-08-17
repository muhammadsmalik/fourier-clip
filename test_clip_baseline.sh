#!/bin/bash
cd DomainBed
echo "Testing CLIPZeroShot on all 4 domains..."

for env in 0 1 2 3; do
    echo "Running test_env $env..."
    python -m domainbed.scripts.train \
        --data_dir ../data \
        --dataset OfficeHome \
        --algorithm CLIPZeroShot \
        --test_env $env \
        --steps 100 \
        --output_dir ../../outputs/clip_baseline_env$env
done

echo "Collecting results..."
python -m domainbed.scripts.collect_results \
    --input_dir ../../outputs/