
# data
OUTPUT_DIR='experiments/data/dice/datasets/v1'
N_SAMPLES=100000

N_OBJECTS='(3,6)' 
OBJECTS_PATTERN='experiments/data/dice/images/dice/**/*' 
BACKGROUNDS_PATTERN='experiments/data/dice/images/backgrounds/*' 
ROTATION_ANGLES=360  
OBJECT_RESIZE=37
BACKGROUND_RESIZE='(512,512)' 
IOU_THRESHOLD=0.0
OBJECT_CHANNEL_MULTIPLY=True
BACKGROUND_CHANNEL_MULTIPLY=True
OBJECT_CHANNEL_INVERT=True
BACKGROUND_CHANNEL_INVERT=False
BACKGROUND_ROTATE=True
OBJECT_SCALE=1.3
OUTPUT_EXTENSION="jpg"

# job
WORKERS=9

# args
for var in "$@"; do

    if [[ "$var" == "--toy" ]]; then
        OUTPUT_DIR='experiments/data/dice/datasets/tmp'
        N_SAMPLES=100
    fi
    if [[ "$var" == "--rm" ]]; then
        REMOVE=true
    fi
    
done

if ${REMOVE:-false} ;then
    rm -fr $OUTPUT_DIR
fi


# run
domain-randomization object-detection \
    --n_samples "$N_SAMPLES" \
    --n_objects "$N_OBJECTS" \
    --objects_pattern "$OBJECTS_PATTERN"  \
    --backgrounds_pattern "$BACKGROUNDS_PATTERN" \
    --output_dir "$OUTPUT_DIR" \
    --rotation_angles "$ROTATION_ANGLES"  \
    --object_resize "$OBJECT_RESIZE" \
    --background_resize "$BACKGROUND_RESIZE" \
    --iou_threshold "$IOU_THRESHOLD" \
    --object_channel_multiply "$OBJECT_CHANNEL_MULTIPLY" \
    --background_channel_multiply "$BACKGROUND_CHANNEL_MULTIPLY" \
    --object_channel_invert "$OBJECT_CHANNEL_INVERT" \
    --background_channel_invert "$BACKGROUND_CHANNEL_INVERT" \
    --background_rotate "$BACKGROUND_ROTATE" \
    --object_scale "$OBJECT_SCALE" \
    --output_extension "$OUTPUT_EXTENSION" \
    --workers "$WORKERS"
