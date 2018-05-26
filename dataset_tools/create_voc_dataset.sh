VOC_root_path="/home/guicunbin/workspace/dataset/VOC2012/"


#   python ../../object_detection/dataset_tools/create_pascal_tf_record.py \
#       --label_map_path=object_detection/data/pascal_label_map.pbtxt \
#       --data_dir="${VOC_root_path}/VOCdevkit/" \
#   	--year=VOC2012 \
#   	--set=train \
#       --output_path="${VOC_root_path}/pascal_train.record"


python ../../object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir="${VOC_root_path}/VOCdevkit/" \
    --year=VOC2012 \
    --set=val \
    --output_path="${VOC_root_path}/pascal_val.record"
