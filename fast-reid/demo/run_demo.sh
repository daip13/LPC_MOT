python demo/visualize_result.py \
    --config-file config/NAIC/sbs_R50-cd.yaml \
    --parallel \
    --vis-label \
    --dataset-name 'NAIC' \
    --output logs/mgn_duke_vis \
    --opts MODEL.WEIGHTS logs/naic/sbs_R50_cd/model_partial.pth
