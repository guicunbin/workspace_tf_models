#coding=utf-8
import cv2

# 打开摄像头
capture = cv2.VideoCapture(0)

while True:
    # 从摄像头读取一帧图像；如果读取失败，ret为False，如果成功，ret为true；frame存储了那一帧的数据。
    ret, frame = capture.read()
    # print(ret)
    # 显示图像
    cv2.imshow("camera", frame)
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
capture.release()
