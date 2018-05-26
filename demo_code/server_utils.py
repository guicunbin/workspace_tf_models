#coding:utf-8
import cv2
import numpy as np
import socket
from PIL import Image
#import Image
import StringIO

encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def create_server_socket(TCP_IP, TCP_PORT):
    '''
    return: sock;
    after get the sock;  You should:
        src, src_addr = sock.accept()
        dst, dst_addr = sock.accept()
        decimg = get_img_from_socket(src)
        decimg = process_image(decimg)
        send_img_to_socket(dst, decimg)
    '''
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #sock = socket.socket()
    sock.bind((str(TCP_IP), int(TCP_PORT)))
    # 注意bind的这里，IP地址和端口号都要与前面的程序中一样
    sock.listen(True)# 监听端口
    return sock


def create_client_socket(TCP_IP, TCP_PORT):
    """
    return: sock;
    alfter get this sock; You should:
        send_img_to_socket(sock, decimg)
        or
        decimg = get_img_from_socket(sock)
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((str(TCP_IP), int(TCP_PORT)))
    return sock






def get_img_from_socket(sock, send_length=16):
    """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        or
        sock, src_addr = sock.accept()
    
    """
    length = recvall(sock,send_length)
    stringData = recvall(sock, int(length))
    data = np.fromstring(stringData, dtype='uint8')
    print "after fromstring data.shape = {}".format(data.shape)
    decimg=cv2.imdecode(data,1)
    print "after imdecode  decimg.shape= {}".format(decimg.shape)
    return decimg


def send_img_to_socket(sock, decimg, send_length=16):
    result, imgencode = cv2.imencode('.jpg', decimg, encode_param)
    data = np.array(imgencode)
    stringData = data.tostring()
    sock.send( str(len(stringData)).ljust(send_length));
    sock.send( stringData );


def get_local_IP():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    local_IP = s.getsockname()[0]
    s.close()
    return local_IP


