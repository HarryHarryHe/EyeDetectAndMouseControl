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

def prepare_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (448, 224))  # 调整图像大小以匹配模型输入
        img = img / 255.0  # 归一化
        return img
    else:
        return None

# 替换 'path_to_your_image.jpg' 为你的图像文件路径
your_image_path = '0_18_Button.left.jpeg'
prepared_image = prepare_image(your_image_path)

# 检查图像是否加载和预处理正确
if prepared_image is not None:
    # 添加一个额外的维度，因为模型期望的输入是一个批量的图像
    prepared_image = np.expand_dims(prepared_image, axis=0)
    print("Image is ready for prediction.")
else:
    print("Failed to prepare the image.")


# 使用准备好的图像进行预测
predictions = model.predict(prepared_image)
print("Predicted coordinates:", predictions)

width, height = 2880, 1800
x, y = predictions[0]
print(x * width, y * height)
pyautogui.moveTo(x * width, y * height, 1)
