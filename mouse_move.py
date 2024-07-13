import cv2
import pyautogui
from gaze_tracking import GazeTracking
import numpy as np

# 初始化摄像头和凝视追踪库
webcam = cv2.VideoCapture(0)
gaze = GazeTracking()

# 获取屏幕尺寸
screenWidth, screenHeight = pyautogui.size()

# 校准点
calibration_points = [(0, 0), (screenWidth, 0), (screenWidth, screenHeight), (0, screenHeight),
                      (screenWidth // 2, screenHeight // 2)]
calibration_data = []


def calibrate():
    for point in calibration_points:
        _, frame = webcam.read()
        gaze.refresh(frame)
        input("请看向屏幕的位置 " + str(point) + " 并按回车继续...")

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        if left_pupil and right_pupil:
            avg_pupil_x = (left_pupil[0] + right_pupil[0]) / 2
            avg_pupil_y = (left_pupil[1] + right_pupil[1]) / 2
            calibration_data.append((avg_pupil_x, avg_pupil_y, point[0], point[1]))


# 进行五点校准
calibrate()

# 计算校准参数
calibration_data = np.array(calibration_data)
x_coeffs = np.linalg.lstsq(calibration_data[:, :2], calibration_data[:, 2], rcond=None)[0]
y_coeffs = np.linalg.lstsq(calibration_data[:, :2], calibration_data[:, 3], rcond=None)[0]


# 映射函数
def map_coords(pupil_x, pupil_y):
    screen_x = pupil_x * x_coeffs[0] + pupil_y * x_coeffs[1]
    screen_y = pupil_x * y_coeffs[0] + pupil_y * y_coeffs[1]
    return int(screen_x), int(screen_y)


# 主循环
while True:
    _, frame = webcam.read()
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    if left_pupil and right_pupil:
        avg_pupil_x = (left_pupil[0] + right_pupil[0]) / 2
        avg_pupil_y = (left_pupil[1] + right_pupil[1]) / 2

        # 映射瞳孔位置到屏幕坐标
        screen_x, screen_y = map_coords(avg_pupil_x, avg_pupil_y)

        # 移动鼠标到新位置
        pyautogui.moveTo(screen_x, screen_y)

    # 显示处理后的视频帧
    cv2.imshow("Gaze Tracking", frame)

    # 按ESC键退出
    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
