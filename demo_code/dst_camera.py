#!/usr/bin/python
import socket
import cv2
import numpy
from server_utils import *
import fire
import time


def main(TCP_IP, TCP_PORT):
    # # #TCP_IP = '10.112.135.94'
    # # TCP_IP = 'localhost'
    # # TCP_PORT = 9999
    # TCP_PORT = int(TCP_PORT)
    # 
    # #sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock = socket.socket()
    # sock.connect((TCP_IP, TCP_PORT))

    sock = create_client_socket(TCP_IP, TCP_PORT)
    
    t1 = time.time()
    while True:

        decimg = get_img_from_socket(sock)

        FPS = 1.0 / (time.time() - t1)
    
        cv2.imshow('From_DST',decimg)
        print "FPS = {}".format(int(FPS))
        if cv2.waitKey(2) == 27: break
        t1 = time.time()
    
    sock.close()
    cv2.destroyAllWindows() 

fire.Fire(main)
