rm -fr experiments/images/results

domain-randomization object-detection \
    --n_samples 20 \
    --n_objects '(3,6)' \
    --objects_pattern 'experiments/images/dice/**/*'  \
    --backgrounds_pattern 'experiments/images/backgrounds/*' \
    --output_dir 'experiments/images/results' \
    --rotation_angles 360  \
    --object_resize 80 \
    --background_resize '(1024,1024)' \
    --iou_threshold 0.0
