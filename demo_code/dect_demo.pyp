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
from matplotlib import pyplot as plt
from PIL import Image
from demo_utils import *
from DC_utils import *


# This is needed since the notebook is stored in the object_detection folder.
TF_MODELS_DIR="/home/guicunbin/workspace/models/"
sys.path.append("{}/research/".format(TF_MODELS_DIR))
sys.path.append("{}/research/object_detection/".format(TF_MODELS_DIR))
from object_detection.utils import ops as utils_ops
# This is needed to display the images.
from utils import label_map_util
from utils import visualization_utils as vis_util
Session_config = tf.ConfigProto()
Session_config.gpu_options.allow_growth = True



sock = create_server_socket(TCP_IP=get_local_IP(), TCP_PORT=9999)
print "wait Destination ...."
dst, dst_addr = sock.accept()
print "Destination Connected by", dst_addr





#model_name="ssd_mobilenet_v2_coco_2018_03_29"
#model_name="mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
#model_name="faster_rcnn_nas_coco_2018_01_28"
#model_name="faster_rcnn_nas_coco_2018_01_28"
#model_name="faster_rcnn_inception_v2_coco_2018_01_28"
model_name="faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
TEST_IMAGE_DIR  = "/home/guicunbin/workspace/dataset/DC_car_detect/test_a/"
graph_path = '/home/guicunbin/workspace/TF_ckpt/{}/frozen_inference_graph.pb'.format(model_name)
PATH_TO_LABELS = ('/home/guicunbin/workspace/models/research/object_detection/data/mscoco_label_map.pbtxt')

f_write = open('/home/guicunbin/workspace/dataset/DC_car_detect/{}_submit_{}_.csv'.format(model_name, get_now_time()), 'w');
f_write.write('name,coordinate\n');


NUM_CLASSES = 90   # mscoco
#NUM_CLASSES = 80    # ava
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



#   detection_graph = tf.Graph()
#   with detection_graph.as_default():
#       od_graph_def = tf.GraphDef()
#       with tf.gfile.GFile(graph_path, 'rb') as fid:
#           serialized_graph = fid.read()
#           od_graph_def.ParseFromString(serialized_graph)
#           tf.import_graph_def(od_graph_def, name='')


detection_graph = restore_graph_from_pbfile(graph_path)



TEST_IMAGE_NAMES= [img_name for img_name in os.listdir(TEST_IMAGE_DIR)]
#image_H = 600; image_W = 600
image_test_0 = cv2.imread("{}/{}".format(TEST_IMAGE_DIR, TEST_IMAGE_NAMES[0]));
image_H, image_W, image_C = image_test_0.shape
print "image_test_0.shape  = {}".format(image_test_0.shape);







with detection_graph.as_default():
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
        if key + ':0'  in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(key + ":0")
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image_H, image_W)
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)


    sess = tf.Session(config = Session_config)
    for image_name in TEST_IMAGE_NAMES:
        image_path = "{}/{}".format(TEST_IMAGE_DIR, image_name);
        # read bgr 3-channel image
        if not image_path.endswith('.jpg'): continue;

        print "image_path = {}".format(image_path)



        #image = cv2.resize(cv2.imread(image_path, 1), (image_H, image_W));
        image = cv2.imread(image_path, 1);
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        print "-->image.shape = {} ".format(image.shape)
        assert len(image.shape) == 3 and image.shape[0]==image_H and image.shape[1] == image_W;
        image_4d = np.expand_dims(image, 0)
        print "-->image_4d.shape = {}".format(image_4d.shape)
        feed_dict={image_tensor: image_4d}
        # Run inference
        t1 = time.time();
        #_ = sess.run(tensor_dict['detection_boxes'], feed_dict=feed_dict)
        output_dict = sess.run(tensor_dict,feed_dict = feed_dict)
        print "----->  using time = {} ms ".format(round((time.time() - t1)*1000, 0));

        #   #print _.shape
        #   output_dict = sess.run(tensor_dict,feed_dict = feed_dict)
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]

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
        #cv2.imwrite("./dect_results.jpg", image)
        send_img_to_socket(dst, image)
        time.sleep(5);
         
        
        DC_car_dect_write_one_line_to_submit(f_write, image_copy, image_name, image_H, image_W, output_dict, 
                dect_cate_list = [3,5,6,7,8,9,62], dect_score_min = 0.5, show_box_on_image=True);
        send_img_to_socket(dst, image_copy)
        time.sleep(5);



        #   DC_car_dect_write_one_line_to_submit(f_write, image, image_name, image_H, image_W, output_dict, 
        #           dect_cate_list = [3,5,6,7,8,9,62], dect_score_min = 0.5, show_box_on_image=False);

sock.close();
dst.close();

