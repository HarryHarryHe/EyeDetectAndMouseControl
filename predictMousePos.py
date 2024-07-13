import numpy as np
import os
import cv2
import dlib
from imutils import face_utils
import pyautogui
from keras.models import load_model


model = load_model("models/conv2d_6-EyeMouseMapper.h5")
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
def scan(image_size=(224, 224)):  # 调整图像大小以更好地匹配模型需求
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)  # 使用dlib检测面部

    if len(rects) == 1:  # 确保检测到一个面部
        rect = rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 提取左眼和右眼的坐标
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # 计算扩展后的眼睛区域
        leftEyeBounds = [max(min(leftEye[:, 0]) - padding, 0),
                         min(max(leftEye[:, 0]) + padding, gray.shape[1]),
                         max(min(leftEye[:, 1]) - padding, 0),
                         min(max(leftEye[:, 1]) + padding, gray.shape[0])]
        rightEyeBounds = [max(min(rightEye[:, 0]) - padding, 0),
                          min(max(rightEye[:, 0]) + padding, gray.shape[1]),
                          max(min(rightEye[:, 1]) - padding, 0),
                          min(max(rightEye[:, 1]) + padding, gray.shape[0])]

        # 从面部坐标提取眼部区域
        leftEyeImage = gray[leftEyeBounds[2]:leftEyeBounds[3], leftEyeBounds[0]:leftEyeBounds[1]]
        rightEyeImage = gray[rightEyeBounds[2]:rightEyeBounds[3], rightEyeBounds[0]:rightEyeBounds[1]]
        leftEyeImage = cv2.resize(leftEyeImage, image_size)
        rightEyeImage = cv2.resize(rightEyeImage, image_size)
        leftEyeImage = normalize(leftEyeImage)
        rightEyeImage = normalize(rightEyeImage)
        if leftEyeImage is not None and rightEyeImage is not None:
            eyes = np.hstack([leftEyeImage, rightEyeImage])
            eyes = np.stack([eyes, eyes, eyes], axis=-1)  # 转换为三通道

            return (eyes * 255).astype(np.uint8)  # 转换类型为 uint8
    else:
        print("Face detected:", len(rects))
        return None

i = 1
while True:
    eyes = scan()
    if not eyes is None:
        # 注意：需要使用 np.expand_dims 增加一个批处理维度，模型预期输入形状为 (1, 224, 448, 3)
        eye = np.expand_dims(eyes, axis=0)
        # if i==1:
        #     cv2.imwrite('eye.jpg', eye)
        #     i=0
        x, y = model.predict(eye,verbose=0)[0]
        print("Mouse position:", x, y)
        pyautogui.moveTo(x * width, y * height, duration=1)
        print("Move mouse to:", x * width, y * height)
