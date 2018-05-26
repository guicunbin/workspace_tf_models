#coding:utf-8
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time
import socket
from server_utils import *
from collections import defaultdict
from io import StringIO
 #from matplotlib import pyplot as plt
from PIL import Image
from demo_utils import *
from DC_utils import *
import fire
# This is needed since the notebook is stored in the object_detection folder.
TF_MODELS_DIR="/home/guicunbin/workspace/models/"
sys.path.append("{}/research/".format(TF_MODELS_DIR))
sys.path.append("{}/research/object_detection/".format(TF_MODELS_DIR))
# This is needed to display the images.
from utils import label_map_util
from utils import visualization_utils as vis_util
Session_config = tf.ConfigProto()
#Session_config.gpu_options.allow_growth = True
Session_config.gpu_options.per_process_gpu_memory_fraction = 0.25



def main(submit_file, log_file, graph_path, label_path, test_image_dir, DC_dect_cate_list=(3,5,6,7,8,9,62), DC_dect_score_min="0.5", is_socked="0"):
    print locals();
    save_params_to_log(**locals());
    is_socked = int(is_socked)
    assert type(DC_dect_cate_list) == tuple
    DC_dect_cate_list = [int(num) for num in DC_dect_cate_list]
    DC_dect_score_min = float(DC_dect_score_min)
    if is_socked:
        sock = create_server_socket(TCP_IP=get_local_IP(), TCP_PORT=9999)
        print "wait Destination ...."
        dst, dst_addr = sock.accept()
        print "Destination Connected by", dst_addr


    
    
    NUM_CLASSES = get_num_classes_from_labeltxt(label_path)
    label_map   = label_map_util.load_labelmap((label_path))
    categories  = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index  = label_map_util.create_category_index(categories)
    
    detection_graph = restore_graph_from_pbfile(graph_path)
    
    test_image_names= [img_name for img_name in os.listdir(test_image_dir)]
    test_image_names.sort();
    #image_H = 600; image_W = 600
    image_test_0 = cv2.imread("{}/{}".format(test_image_dir, test_image_names[0]));
    image_H, image_W, image_C = image_test_0.shape
    print "image_test_0.shape  = {}".format(image_test_0.shape);
    


    is_continue_to_run = os.path.exists(submit_file)


    f_write = open(submit_file, 'a+');
    head_line = 'name,coordinate\n'
    if not is_continue_to_run:
        f_write.write(head_line);
    else:
        f_submit_lines = f_write.readlines();
        assert f_submit_lines[0] == head_line
        match_index=len(f_submit_lines)-2
        assert f_submit_lines[-1].split('.jpg')[0]==test_image_names[match_index].split('.jpg')[0]
        print "\n\n---> continue write to {} : line_{}\n\n".format(submit_file, len(f_submit_lines))
        split_string="-----------------------continue----------------------------\n";   f_write.write(split_string);
        test_image_names = test_image_names[match_index:]
        print "\n remain number = {}".format(len(test_image_names)) 


    
    with detection_graph.as_default():
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
            if key + ':0'  in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(key + ":0")
        if 'detection_masks' in tensor_dict:
            expand_this_graph_by_process_detection_masks(tensor_dict)
    
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        sess         = tf.Session(config = Session_config)
        for image_name in test_image_names:
            image_path = "{}/{}".format(test_image_dir, image_name);
            if not image_path.endswith('.jpg'): continue;
            print "image_path = {}".format(image_path)
            #image   = cv2.resize(cv2.imread(image_path, 1), (image_H, image_W));
            image    = cv2.imread(image_path, 1);
            image_4d = np.expand_dims(image, 0);
            assert len(image_4d.shape)==4 and image_4d.shape[1]==image_H and image_4d.shape[2] == image_W;
            print "-->image_4d.shape = {}".format(image_4d.shape)
            feed_dict={image_tensor: image_4d}
            # Run inference
            t1 = time.time();
            output_dict = sess.run(tensor_dict,feed_dict = feed_dict)
            print "----->  using time = {} ms ".format(round((time.time() - t1)*1000, 0));


            if is_socked:
                np.savez("./output_dict", **output_dict); 
            convert_output_dict_to_suitable_dtype(output_dict);
            
            if is_socked:
                image_copy = image.copy()
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=8)
                send_img_to_socket(dst, image)
                time.sleep(5);
                 
                DC_car_dect_write_one_line_to_submit(f_write, image_copy, image_name, image_H, image_W, output_dict, 
                        dect_cate_list = DC_dect_cate_list, dect_score_min = DC_dect_score_min, show_box_on_image=True);
                send_img_to_socket(dst, image_copy)
                time.sleep(5);
            else:
                DC_car_dect_write_one_line_to_submit(f_write, image, image_name, image_H, image_W, output_dict, 
                        dect_cate_list = DC_dect_cate_list, dect_score_min = DC_dect_score_min, show_box_on_image=False);

    if is_socked:
        sock.close();
        dst.close();

if __name__ == "__main__":
    fire.Fire(main)
