#coding:utf-8
import numpy
import socket
import time
from server_utils import *
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import ast


#   #model = 'cmu' ## mobilenet_thin
#   model = 'mobilenet_thin' ## mobilenet_thin
#   w, h = 432, 368
#   scales = '[1.0]'
#   
#   e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))


def process_image(image):
    humans = e.inference(image, scales=scales)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    return image

##  #TCP_IP = 'localhost'
##  TCP_IP = '10.112.135.94'
##  TCP_PORT = 9999
##  
##  #sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
##  sock = socket.socket()
##  sock.bind((TCP_IP, TCP_PORT))
##  # 注意bind的这里，IP地址和端口号都要与前面的程序中一样
##  sock.listen(True)# 监听端口

sock = create_server_socket(TCP_IP=get_local_IP(), TCP_PORT=9999)

# 等待数据源端连接
print "wait Source .... "
src, src_addr = sock.accept()
print "Source Connected by", src_addr

# 等待目标端连接
print "wait Destination ...."
dst, dst_addr = sock.accept()
print "Destination Connected by", dst_addr

while True:
    decimg = get_img_from_socket(src)

    decimg=process_image(decimg)

    send_img_to_socket(dst, decimg)


src.close()
dst.close()
sock.close()
