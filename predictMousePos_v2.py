import numpy as np
import os
import cv2
import dlib
from imutils import face_utils
import pyautogui
from keras.models import load_model

model = load_model("models/conv2d_2-EyeMouseMapper-0708.h5")
width, height = 2880, 1800
# 初始化dlib的面部检测器和面部标记预测器
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor_path = "models/shape_predictor_68_face_landmarks.dat"  # 这里填写预测器文件路径
predictor = dlib.shape_predictor(predictor_path)
cascade = cv2.CascadeClassifier("models/haarcascade_eye.xml")
video_capture = cv2.VideoCapture(0)
padding = 10  # 眼睛区域扩展的像素数


def normalize(x):
    minn, maxx = x.min(), x.max()
    return (x - minn) / (maxx - minn)


# TODO 图片大小需要修改
def scan(image_size=(448, 224)):
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) == 1:
        rect = rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        x_min = min(leftEye[:, 0].min(), rightEye[:, 0].min()) - padding
        x_max = max(leftEye[:, 0].max(), rightEye[:, 0].max()) + padding
        y_min = min(leftEye[:, 1].min(), rightEye[:, 1].min()) - padding
        y_max = max(leftEye[:, 1].max(), rightEye[:, 1].max()) + padding

        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, gray.shape[1])
        y_max = min(y_max, gray.shape[0])

        eyeRegion = gray[y_min:y_max, x_min:x_max]
        eyeRegion = cv2.resize(eyeRegion, image_size)
        eyeRegion = normalize(eyeRegion)
        return (eyeRegion * 255).astype(np.uint8)
    else:
        print("Face detected:", len(rects))
        return None


i = 1
while True:
    eyes = scan()
    if not eyes is None:
        # 注意：需要使用 np.expand_dims 增加一个批处理维度，模型预期输入形状为 (1, 224, 448, 3)
        eye = cv2.resize(eyes, (448, 224))  # 调整图像大小以匹配模型输入
        eye = eye / 255.0  # 归一化像素值
        eye = np.expand_dims(eye, axis=-1)  # 增加一个维度以匹配模型输入要求
        eye = np.expand_dims(eye, axis=0)  # 增加批处理维度
        # if i==1:
        #     cv2.imwrite('eye.jpg', eye)
        #     i=0
        x, y = model.predict(eye, verbose=0)[0]
        print("Mouse position:", x, y)
        pyautogui.moveTo(x, y, duration=1)
        print("Move mouse to:", x, y)
