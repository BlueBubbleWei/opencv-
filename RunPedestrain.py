import numpy as np
import cv2
from Pedestrian import *
def main():
    # 加载视频
    # camera = cv2.VideoCapture('../movie.mpg')
    camera = cv2.VideoCapture('Lara_UrbSeq1_CAOR_v3_7_HR.avi')
    # camera = cv2.VideoCapture(r'D:\opencv\opencv\sources\samples\data\768x576.avi')
    # 初始化背景分割器
    history = 20
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    # 创建显示主窗口
    cv2.namedWindow('surveillance')
    pedestrians = {}  # 行人字典
    firstFrame = True
    frames = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../output.avi', fourcc, 20.0, (640, 480))

    while (True):
        print('----------------------frmae %d----------------' % frames)
        grabbed, frane = camera.read()
        if (grabbed is False):
            print("failed to grab frame")
            break
        ret, frame = camera.read()
        fgmask = bs.apply(frame)

        if frames < history:
            frames += 1
            continue
        # 设置阈值
        th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]
        # 腐蚀
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        # 膨胀
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        # 查找轮廓
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter = 0
        for c in contours:
            if cv2.contourArea(c) > 500:
                # 边界数组
                (x, y, w, h) = cv2.boundingRect(c)
                # 绘制矩形
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                if firstFrame is True:
                    pedestrians[counter] = Pedestrian(counter, frame, (x, y, w, h))
                counter += 1
        # 更新帧内容
        for i, p in pedestrians.items():
            p.update(frame)

        # false 只跟踪已有的行人
        # firstFrame = True
        firstFrame = False
        frames += 1

        # 显示
        cv2.imshow('surveillance', frame)
        out.write(frame)
        if cv2.waitKey(120) & 0xFF == 27:  # esc退出
            break
    out.release()
    camera.release()


if __name__ == "__main__":
    main()