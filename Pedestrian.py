import cv2
import numpy as np
import os.path as path
import argparse

font = cv2.FONT_HERSHEY_SIMPLEX

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm",
                    help="m (or nothing) for meanShift and c for camshift")
args = vars(parser.parse_args())


# 计算矩阵中心(行人位置)
def center(points):
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    # print(np.array([np.float32(x), np.float32(y)], np.float32))
    # [ 588.   257.5]
    return np.array([np.float32(x), np.float32(y)], np.float32)


# 行人
class Pedestrian():
    def __init__(self, id, frame, track_window):
        self.id = int(id)  # 行人id
        x, y, w, h = track_window  # 跟踪窗体
        self.track_window = track_window
        # 更换颜色空间
        self.roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
        # 计算roi图形的彩色直方图
        roi_hist = cv2.calcHist([self.roi], [0], None, [16], [0, 180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # 设置卡尔曼滤波器
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               np.float32) * 0.03
        # 测量坐标
        self.measurement = np.array((2, 1), np.float32)
        # 鼠标运动预测
        self.predication = np.zeros((2, 1), np.float32)
        # 指定停止条件
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.center = None
        self.update(frame)

    def __del__(self):
        print('Pedestrian %d destroyed' % self.id)

    # 更新图像帧
    def update(self, frame):
        # 更换颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # histogram back projection calculation 直方图反向投影
        back_project = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        # camshift
        if args.get('algorithm') == 'c':
            ret, self.track_window = cv2.CamShift(back_project, self.track_window, self.term_crit)
            # 绘制跟踪框
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            self.center = center(pts)
            cv2.polylines(frame, [pts], True, 255, 1)

        # 均值漂移
        if not args.get('algorithm') or args.get('algorithm') == 'm':
            ret, self.track_window = cv2.meanShift(back_project, self.track_window, self.term_crit)
            # 绘制跟踪框
            x, y, w, h = self.track_window
            self.center = center([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        self.kalman.correct(self.center)
        prediction = self.kalman.predict()
        cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 4, (0, 255, 0), -1)
        # 计数器
        cv2.putText(frame, 'ID: %d --> %s' % (self.id, self.center), (11, (self.id + 1) * 25 + 1), font, 0.6, (0, 0, 0),
                    1, cv2.LINE_AA)
        # 跟踪窗口坐标
        cv2.putText(frame, 'ID: %d --> %s' % (self.id, self.center), (10, (self.id + 1) * 25), font, 0.6, (0, 255, 0),
                    1, cv2.LINE_AA)
