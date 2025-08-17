#!/bin/bash
cd DomainBed
echo "Testing FADA-CLIP on all 4 domains..."

for env in 0 1 2 3; do
    echo "Running FADA-CLIP test_env $env..."
    python -m domainbed.scripts.train \
        --data_dir ../data \
        --dataset OfficeHome \
        --algorithm FADA_CLIP \
        --test_env $env \
        --steps 100 \
        --output_dir ../../outputs/fada_clip_env$env
done

echo "Collecting results..."
python -m domainbed.scripts.collect_results \
    --input_dir ../../outputs/