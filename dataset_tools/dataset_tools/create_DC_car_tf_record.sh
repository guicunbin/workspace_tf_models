dataset_root="/home/guicunbin/workspace/dataset/DC_car_detect/"
TRAIN_IMAGE_DIR="${dataset_root}/train/"
VAL_IMAGE_DIR=${TRAIN_IMAGE_DIR}
TRAIN_ANNOTATIONS_FILE="${dataset_root}/train_1w.csv"
VAL_ANNOTATIONS_FILE="${dataset_root}/val_1000.csv"
OUTPUT_DIR="${dataset_root}/"
python ./create_DC_car_tf_record.py \
    --logtostderr \
    --include_masks \
    --train_image_dir="${TRAIN_IMAGE_DIR}" \
    --val_image_dir="${VAL_IMAGE_DIR}" \
    --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
    --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
    --output_dir="${OUTPUT_DIR}"


