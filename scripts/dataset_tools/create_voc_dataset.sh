TF_models_path="/home/guicunbin/workspace/models/"
VOC_root_path="/home/guicunbin/workspace/dataset/VOC2012/"
label_map_path="/home/guicunbin/workspace/workspace_tf_models/label_pbtxt/pascal_label_map.pbtxt"


python ${TF_models_path}/research/object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=${label_map_path} \
    --data_dir="${VOC_root_path}/VOCdevkit/" \
	--year=VOC2012 \
	--set=train \
    --output_path="${VOC_root_path}/pascal_train.record"


python ${TF_models_path}/research/object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=${label_map_path} \
    --data_dir="${VOC_root_path}/VOCdevkit/" \
    --year=VOC2012 \
    --set=val \
    --output_path="${VOC_root_path}/pascal_val.record"
