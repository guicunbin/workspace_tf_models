pets_root_path="/home/guicunbin/workspace/dataset/pets/"
python ../../object_detection/dataset_tools/create_pet_tf_record.py \
    --label_map_path=object_detection/data/pet_label_map.pbtxt \
    --data_dir=${pets_root_path} \
    --output_dir=${pets_root_path}
