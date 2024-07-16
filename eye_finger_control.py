import argparse
import time
import dlib
import cv2
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream
import tensorflow as tf
from keras.models import load_model
import pyautogui
import mediapipe as mp

# 构造参数解析并解析参数
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True, help="路径到面部标记预测器")
# ap.add_argument("-m", "--model", required=True, help="路径到眼部状态预测模型")
# args = vars(ap.parse_args())

# 初始化dlib的面部检测器（基于HOG）和面部标记预测器
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
model = load_model('models/efficientnetb0-EyeDetection-92.83.h5')

# 初始化MediaPipe手势识别
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

# 初始化MediaPipe绘图工具
mp_drawing = mp.solutions.drawing_utils

# 开始从网络摄像头获取视频流
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

padding = 10  # 设置扩展像素数，方便获取更多眼部区域用于模型预测

# 初始化一些用于检测眼睛状态的变量
left_eye_status = "open"
right_eye_status = "open"
left_blink_time = 0
right_blink_time = 0
blink_threshold = 0.5
both_eyes_blinked = False
left_holding = False
right_holding = False

# 获取屏幕尺寸
screen_width, screen_height = pyautogui.size()

# 获取视频帧尺寸
frame = vs.read()
if frame is None:
    print("Failed to read frame from video stream.")
    vs.stop()
    cv2.destroyAllWindows()
    exit()
frame_height, frame_width, _ = frame.shape

# 计算视频帧中心区域的边界
center_x, center_y = frame_width // 2, frame_height // 2
center_area_width, center_area_height = 128, 64  # 中心监控区域的一半宽度和高度

control_active = False  # 控制鼠标的激活状态

# 平滑滤波器初始化
last_x, last_y = screen_width // 2, screen_height // 2
smooth_factor = 5


# 模型预测
def predict_eye_status(eyes, model, eye_image):
    gray_image = np.stack((eye_image,) * 3, axis=-1)
    gray_image = cv2.resize(gray_image, (224, 224))
    gray_image = np.expand_dims(gray_image, axis=0)
    predictions = model.predict(gray_image, verbose=0)
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class = ['open' if predicted_class_index[0] == 1 else 'closed']
    return predicted_class[0]


# 检测眼睛是否眨眼
def check_eye_blink(eye, status, previous_status, previous_time, is_holding):
    current_time = time.time()
    if status == "closed" and previous_status == "open":
        return "closed", current_time, is_holding
    elif status == "open" and previous_status == "closed":
        if current_time - previous_time < 0.37:
            if not is_holding:
                if eye == "left":
                    pyautogui.click(button='left')
                    print("Left eye clicked, click left button")
                elif eye == "right":
                    pyautogui.click(button='right')
                    print("Right eye clicked, click right button")
        elif is_holding:
            if eye == "left":
                pyautogui.mouseUp(button='left')
                print("Left eye open, release left button")
            elif eye == "right":
                pyautogui.mouseUp(button='right')
                print("Right eye open, release right button")
        return "open", current_time, False
    elif status == "closed" and previous_status == "closed" and is_holding:
        return "closed", previous_time, is_holding
    elif status == "closed" and previous_status == "closed" and not is_holding:
        if current_time - previous_time > 0.5:
            is_holding = True
            if eye == "left":
                pyautogui.mouseDown(button='left')
                print("Left eye closed, hold left button")
            elif eye == "right":
                pyautogui.mouseDown(button='right')
                print("Right eye closed, hold right button")
        return "closed", previous_time, is_holding
    return previous_status, previous_time, is_holding


# 从视频流循环帧
while True:
    frame = vs.read()
    if frame is None:
        print("Failed to read frame from video stream.")
        vs.stop()
        cv2.destroyAllWindows()
        exit()
    # 水平翻转图像
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 在视频帧中心绘制红色圆形标记
    new_center_y = frame_height - center_area_height  # 底部中心的y坐标
    cv2.circle(frame, (center_x, new_center_y), 10, (0, 0, 255), -1)  # 在底部中心绘制红点

    # 将绿框移动到底部中心
    new_center_y = frame_height - center_area_height  # 更新中心y坐标到底部
    cv2.rectangle(frame, (center_x - center_area_width, new_center_y - center_area_height),
                  (center_x + center_area_width, new_center_y + center_area_height), (0, 255, 0), 2)

    # # 绘制中心监控区域的绿色矩形框
    # cv2.rectangle(frame, (center_x - center_area_width, center_y - center_area_height),
    #               (center_x + center_area_width, center_y + center_area_height), (0, 255, 0), 2)

    # 在灰度帧中检测面孔
    rects = detector(gray, 0)

    # 处理手势
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 获取食指尖端的坐标
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(tip.x * frame_width), int(tip.y * frame_height)

            # 检查手指是否在中心监控区域内
            if (center_x - center_area_width <= x <= center_x + center_area_width) and (
                    new_center_y - center_area_height <= y <= new_center_y + center_area_height):
                if not control_active:
                    control_active = True
                    pyautogui.moveTo(screen_width / 2, screen_height / 2)

                # 映射手指的位置到整个屏幕
                screen_x = np.interp(x, [center_x - center_area_width, center_x + center_area_width], [0, screen_width])
                screen_y = np.interp(y, [new_center_y - center_area_height, new_center_y + center_area_height], [0, screen_height])


                # 应用平滑滤波
                screen_x = (last_x * (smooth_factor - 1) + screen_x) / smooth_factor
                screen_y = (last_y * (smooth_factor - 1) + screen_y) / smooth_factor
                pyautogui.moveTo(screen_x, screen_y)

                last_x, last_y = screen_x, screen_y  # 更新上一次的坐标

            # 绘制手部关键点和连线
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 遍历检测到的面孔
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 提取左眼和右眼的坐标
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # 检查并调整眼部坐标
        if leftEye[:, 0].mean() > rightEye[:, 0].mean():  # 如果左眼坐标在右眼坐标右边
            leftEye, rightEye = rightEye, leftEye  # 交换左右眼坐标

        # 计算扩展后的眼睛区域
        leftEyeBounds = [max(min(leftEye[:, 0]) - padding, 0),
                         min(max(leftEye[:, 0]) + padding, gray.shape[1]),
                         max(min(leftEye[:, 1]) - padding, 0),
                         min(max(leftEye[:, 1]) + padding, gray.shape[0])]
        rightEyeBounds = [max(min(rightEye[:, 0]) - padding, 0),
                          min(max(rightEye[:, 0]) + padding, gray.shape[1]),
                          max(min(rightEye[:, 1]) - padding, 0),
                          min(max(rightEye[:, 1]) + padding, gray.shape[0])]

        # 根据扩展的坐标裁剪眼睛区域
        leftEyeImg = gray[leftEyeBounds[2]:leftEyeBounds[3], leftEyeBounds[0]:leftEyeBounds[1]]
        rightEyeImg = gray[rightEyeBounds[2]:rightEyeBounds[3], rightEyeBounds[0]:rightEyeBounds[1]]

        # 确保眼睛区域图像不为空
        if leftEyeImg.size > 0 and rightEyeImg.size > 0:
            # 使用模型预测眼睛状态
            left_status = predict_eye_status("Left", model, leftEyeImg)
            right_status = predict_eye_status("Right", model, rightEyeImg)

            if left_status == 'closed' and right_status == 'closed':
                both_eyes_blinked = True
                print("Both eyes blinked simultaneously")
            else:
                both_eyes_blinked = False

            if not both_eyes_blinked:
                left_eye_status, left_blink_time, left_holding = check_eye_blink("left", left_status, left_eye_status,
                                                                                 left_blink_time, left_holding)
                right_eye_status, right_blink_time, right_holding = check_eye_blink("right", right_status,
                                                                                    right_eye_status, right_blink_time,
                                                                                    right_holding)
        else:
            left_status = 'Unknown'
            right_status = 'Unknown'

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # 按下"q"键，退出循环
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
