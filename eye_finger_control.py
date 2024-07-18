import argparse
import time
import dlib
import cv2
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream
import keras
from keras.models import load_model
import pyautogui
import mediapipe as mp

# 初始化dlib的面部检测器（基于HOG）和面部标记预测器
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
model = load_model('models/efficientnetb0-EyeDetection-92.83.h5')

# 初始化MediaPipe手势识别
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 开始从网络摄像头获取视频流
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

EYE_EXTRACT_PADDING = 10  # 设置扩展像素数，方便获取更多眼部区域用于模型预测
RECTANGLE_PADDING_TOP = 50  # 设置矩形区域顶部扩展像素数

BLINK_THRESHOLD_TIME = 0.37  # 眨眼阈值，该值内判定为眨眼
BLINK_CLOSED_TIME = 0.5  # 闭眼持续时间阈值，该值内判定为长按鼠标

CONTROL_AREA_WIDTH_PADDING, CONTROL_AREA_HEIGHT_PADDING = 75, 75  # 设置控制区域边界扩展像素数，便于鼠标边缘控制，方便映射

CONTROL_AREA_WIDTH, CONTROL_AREA_HEIGHT = 160, 80

CONTROL_ACTIVE = False  # 控制鼠标的激活状态

# 初始化一些用于检测眼睛状态的变量
LEFT_EYE_STATUS, RIGHT_EYE_STATUS = "open", "open"
LEFT_BLINK_TIME, RIGHT_BLINK_TIME = 0, 0
BOTH_EYES_BLINKED, LEFT_HOLDING, RIGHT_HOLDING = False, False, False

# 获取屏幕尺寸
screen_width, screen_height = pyautogui.size()
print("Screen size: ", screen_width, screen_height)

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

# 平滑滤波器初始化
last_x, last_y = screen_width // 2, screen_height // 2
SMOOTH_FACTOR = 3  # 平滑滤波器系数


# 模型预测
def predict_eye_status(eye_image):
    gray_image = np.stack((eye_image,) * 3, axis=-1)
    gray_image = cv2.resize(gray_image, (224, 224))
    gray_image = np.expand_dims(gray_image, axis=0)
    predictions = model.predict(gray_image, verbose=0)
    predicted_class_index = np.argmax(predictions, axis=1)
    return 'open' if predicted_class_index[0] == 1 else 'closed'


# 检测眼睛是否眨眼
def check_eye_blink(eye, status, previous_status, previous_time, is_holding):
    current_time = time.time()
    # 眼睛状态从开变成闭，开始计时
    if status == "closed" and previous_status == "open":
        return "closed", current_time, is_holding

    # 眼睛状态从闭变为开，检查闭合时长
    elif status == "open" and previous_status == "closed":
        # 检查闭合时长以决定是点击还是长按
        if current_time - previous_time < BLINK_THRESHOLD_TIME:  # 假设小于0.37秒为点击, ""为长按经过测试0.37最符合我个人的操作习惯""
            if not is_holding:
                if eye == "left":
                    pyautogui.click(button='left')  # 执行单击操作
                    print("Left eye clicked, click left button")
                elif eye == "right":
                    pyautogui.click(button='right')
                    print("Right eye clicked, click right button")
        elif is_holding:
            if eye == "left":
                pyautogui.mouseUp(button='left')  # 释放鼠标左键
                print("Left eye open, release left button")
            elif eye == "right":
                pyautogui.mouseUp(button='right')  # 释放鼠标右键
                print("Right eye open, release right button")
        return "open", current_time, False

    # 如果眼睛持续闭合并且已经长按 不做操作
    elif status == "closed" and previous_status == "closed" and is_holding:
        # 眼睛持续闭合，但不再次按下鼠标，只更新时间
        return "closed", previous_time, is_holding

    # 眼睛持续闭合但还未触发长按（超过某个阈值，例如0.5秒）
    elif status == "closed" and previous_status == "closed" and not is_holding:
        if current_time - previous_time > BLINK_CLOSED_TIME:
            is_holding = True
            if eye == "left":
                pyautogui.mouseDown(button='left')  # 模拟鼠标左键按下
                print("Left eye closed, hold left button")
            elif eye == "right":
                pyautogui.mouseDown(button='right')  # 模拟鼠标右键按下
                print("Right eye closed, hold right button")
        return "closed", previous_time, is_holding

    # 状态没有变化，返回原状态
    return previous_status, previous_time, is_holding


# 从视频流循环帧
while True:
    frame = vs.read()
    if frame is None:
        print("Failed to read frame from video stream.")
        vs.stop()
        cv2.destroyAllWindows()
        exit()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    new_center_y = CONTROL_AREA_HEIGHT // 2 + RECTANGLE_PADDING_TOP  # 将中心区域移动到顶部区域
    cv2.circle(frame, (center_x, new_center_y), 10, (0, 0, 255), -1)
    cv2.rectangle(frame, (center_x - CONTROL_AREA_WIDTH, new_center_y - CONTROL_AREA_HEIGHT),
                  (center_x + CONTROL_AREA_WIDTH, new_center_y + CONTROL_AREA_HEIGHT), (0, 255, 0), 2)

    rects = detector(gray, 0)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(tip.x * frame_width), int(tip.y * frame_height)

            if (center_x - CONTROL_AREA_WIDTH <= x <= center_x + CONTROL_AREA_WIDTH) and (
                    new_center_y - CONTROL_AREA_HEIGHT <= y <= new_center_y + CONTROL_AREA_HEIGHT):
                if not CONTROL_ACTIVE:
                    CONTROL_ACTIVE = True
                    pyautogui.moveTo(screen_width / 2, screen_height / 2)

                screen_x = np.interp(x, [center_x - CONTROL_AREA_WIDTH, center_x + CONTROL_AREA_WIDTH],
                                     [0 - CONTROL_AREA_WIDTH_PADDING, screen_width + CONTROL_AREA_WIDTH_PADDING])
                screen_y = np.interp(y, [new_center_y - CONTROL_AREA_HEIGHT, new_center_y + CONTROL_AREA_HEIGHT],
                                     [0 - CONTROL_AREA_HEIGHT_PADDING, screen_height + CONTROL_AREA_HEIGHT_PADDING])

                screen_x = (last_x * (SMOOTH_FACTOR - 1) + screen_x) / SMOOTH_FACTOR
                screen_y = (last_y * (SMOOTH_FACTOR - 1) + screen_y) / SMOOTH_FACTOR
                pyautogui.moveTo(screen_x, screen_y)

                last_x, last_y = screen_x, screen_y

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye, rightEye = shape[lStart:lEnd], shape[rStart:rEnd]

        if leftEye[:, 0].mean() > rightEye[:, 0].mean():
            leftEye, rightEye = rightEye, leftEye

        leftEyeBounds = [max(min(leftEye[:, 0]) - EYE_EXTRACT_PADDING, 0),
                         min(max(leftEye[:, 0]) + EYE_EXTRACT_PADDING, gray.shape[1]),
                         max(min(leftEye[:, 1]) - EYE_EXTRACT_PADDING, 0),
                         min(max(leftEye[:, 1]) + EYE_EXTRACT_PADDING, gray.shape[0])]
        rightEyeBounds = [max(min(rightEye[:, 0]) - EYE_EXTRACT_PADDING, 0),
                          min(max(rightEye[:, 0]) + EYE_EXTRACT_PADDING, gray.shape[1]),
                          max(min(rightEye[:, 1]) - EYE_EXTRACT_PADDING, 0),
                          min(max(rightEye[:, 1]) + EYE_EXTRACT_PADDING, gray.shape[0])]

        leftEyeImg = gray[leftEyeBounds[2]:leftEyeBounds[3], leftEyeBounds[0]:leftEyeBounds[1]]
        rightEyeImg = gray[rightEyeBounds[2]:rightEyeBounds[3], rightEyeBounds[0]:rightEyeBounds[1]]

        if leftEyeImg.size > 0 and rightEyeImg.size > 0:
            left_status = predict_eye_status(leftEyeImg)
            right_status = predict_eye_status(rightEyeImg)

            if left_status == 'closed' and right_status == 'closed':
                BOTH_EYES_BLINKED = True
                print("Both eyes blinked simultaneously")
            else:
                BOTH_EYES_BLINKED = False

            if not BOTH_EYES_BLINKED:
                LEFT_EYE_STATUS, LEFT_BLINK_TIME, LEFT_HOLDING = check_eye_blink("left", left_status, LEFT_EYE_STATUS,
                                                                                 LEFT_BLINK_TIME, LEFT_HOLDING)
                RIGHT_EYE_STATUS, RIGHT_BLINK_TIME, RIGHT_HOLDING = check_eye_blink("right", right_status,
                                                                                    RIGHT_EYE_STATUS, RIGHT_BLINK_TIME,
                                                                                    RIGHT_HOLDING)
        else:
            left_status, right_status = 'Unknown', 'Unknown'

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
