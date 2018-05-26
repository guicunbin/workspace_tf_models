#!/usr/bin/python
import socket
import cv2
import numpy
from server_utils import *
import fire

def main(TCP_IP, TCP_PORT):
    # #TCP_IP = '10.112.135.94'
    # # TCP_IP = 'localhost'
    # # TCP_PORT = 9999
    # TCP_PORT = int(TCP_PORT)
    # 
    # sock = socket.socket()
    # sock.connect((TCP_IP, TCP_PORT))

    sock = create_client_socket(TCP_IP, TCP_PORT)
    
    capture = cv2.VideoCapture(0)
    #   capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    #   capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 600)
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 600)
    
    ret, frame = capture.read()
    while(ret):

        #print "type(frame)= {}".format(type(frame))
        send_img_to_socket(sock, frame)
    	
    	ret, frame = capture.read()
    
    sock.close()
    cv2.destroyAllWindows()

fire.Fire(main)
