source ./run_dependency.sh
#---> find the latest_num for ckpt
for fi in $(ls -t $(find ${PATH_TO_TRAIN_DIR} -name "*model.ckpt*"));
do 
    latest_num=$(echo $fi | awk -F'.meta' '{print $1}' | awk -F'.ckpt-' '{print $2}'); 
    break; 
done;
trained_checkpoint_prefix=${PATH_TO_TRAIN_DIR}/model.ckpt-${latest_num}
echo ${PATH_TO_YOUR_PIPELINE_CONFIG}
echo ${trained_checkpoint_prefix}
echo ${EXPORT_DIR}



CUDA_VISIBLE_DEVICES=1 python ${TF_models_path}/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --trained_checkpoint_prefix ${trained_checkpoint_prefix} \
    --output_directory ${EXPORT_DIR}
