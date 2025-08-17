#!/bin/bash
cd DomainBed
echo "Quick FADA-CLIP functionality test..."

python -m domainbed.scripts.train \
    --data_dir ../data \
    --dataset OfficeHome \
    --algorithm FADA_CLIP \
    --test_env 0 \
    --steps 10 \
    --output_dir ../../outputs/quick_test