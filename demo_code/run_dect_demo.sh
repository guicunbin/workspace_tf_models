#CUDA_VISIBLE_DEVICES=1 python ./dect_demo_by_src_and_dst.py
#CUDA_VISIBLE_DEVICES=1 python ./dect_demo_by_IpCamera_and_dst.py


Date_suffix=$(date '+%Y_%m_%d_%H_%M_%S')
dataset_root="/home/guicunbin/workspace/dataset/DC_car_detect/"
#submit_file="${dataset_root}/submit/submit_${Date_suffix}.csv"
submit_file="${dataset_root}/submit/submit_2018_05_26_11_47_37.csv"
log_file="${dataset_root}/run_log.txt"
graph_path="/home/guicunbin/workspace/workspace_tf_models/models/faster_rcnn_inception_resnet_v2_atrous_DC_car/inference/frozen_inference_graph.pb"
test_image_dir="/home/guicunbin/workspace/dataset/DC_car_detect/test_a/"
label_path='/home/guicunbin/workspace/models/research/object_detection/data/mscoco_label_map.pbtxt'
DC_dect_cate_list="3,5,6,7,8,9,62"
DC_dect_score_min="0.5"
is_socked="0"
CUDA_VISIBLE_DEVICES=1 python ./dect_demo.py \
    --submit_file ${submit_file} \
    --log_file ${log_file} \
    --graph_path ${graph_path} \
    --label_path ${label_path} \
    --test_image_dir ${test_image_dir} \
    --DC_dect_cate_list ${DC_dect_cate_list} \
    --DC_dect_score_min ${DC_dect_score_min} \
    --is_socked ${is_socked}





