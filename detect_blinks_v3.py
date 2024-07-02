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

# 构造参数解析并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="路径到面部标记预测器")
ap.add_argument("-m", "--model", required=True, help="路径到眼部状态预测模型")
args = vars(ap.parse_args())

# 初始化dlib的面部检测器（基于HOG）和面部标记预测器
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
model = load_model(args["model"])

# 开始从网络摄像头获取视频流
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
i = 1
padding = 10  # 设置扩展像素数，方便获取更多眼部区域用于模型预测

# 初始化一些用于检测眼睛状态的变量
left_eye_status = "open"  # 初始状态设为开
right_eye_status = "open"
left_blink_time = 0
right_blink_time = 0
blink_threshold = 0.5  # 设置眨眼识别的时间阈值（秒）
both_eyes_blinked = False  # 标志变量来记录双眼是否同时眨眼
# 在主循环中添加和更新这两个变量
left_is_mouse_down = False
right_is_mouse_down = False


# 模型加载
def load_the_model(model_path):
    # models/efficientnetb0-EyeDetection-92.83.h5
    return load_model(model_path)  # 加载模型


# 模型预测
def predict_eye_status(eyes, model, eye_image):
    # 将灰度图像扩展为三通道
    gray_image = np.stack((eye_image,) * 3, axis=-1)

    # 调整图像大小以匹配模型的输入尺寸
    gray_image = cv2.resize(gray_image, (224, 224))

    # 为预测增加批量维度
    gray_image = np.expand_dims(gray_image, axis=0)
    # 使用模型预测眼睛状态
    predictions = model.predict(gray_image, verbose=0)  # verbose=0不显示进度条
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class = ['open' if predicted_class_index[0] == 1 else 'closed']
    # 打印眼睛状态
    # print(f"Predicted {eyes} eye status: {predicted_class}")
    return predicted_class[0]


# 检测眼睛是否眨眼并控制鼠标按下和释放
def check_eye_blink(eye, status, previous_status, previous_time, is_mouse_down):
    current_time = time.time()
    if status != previous_status:
        if status == "closed":
            # 眼睛闭上时更新时间
            return "closed", current_time, is_mouse_down
        elif status == "open" and previous_status == "closed":
            if (current_time - previous_time) <= blink_threshold:
                # 检测到眨眼
                if is_mouse_down:
                    # 如果鼠标已经按下，现在释放
                    if eye == "left":
                        pyautogui.mouseUp(button='left')
                    elif eye == "right":
                        pyautogui.mouseUp(button='right')
                    print(f"{eye.capitalize()} eye blink detected and mouse button released")
                    is_mouse_down = False
                else:
                    # 如果鼠标未按下，现在按下
                    if eye == "left":
                        pyautogui.mouseDown(button='left')
                    elif eye == "right":
                        pyautogui.mouseDown(button='right')
                    print(f"{eye.capitalize()} eye blink detected and mouse button pressed")
                    is_mouse_down = True
            return "open", current_time, is_mouse_down
    return previous_status, previous_time, is_mouse_down


# 从视频流循环帧
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 在灰度帧中检测面孔
    rects = detector(gray, 0)

    # 遍历检测到的面孔
    for rect in rects:
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
                # 检查左眼是否有意义的眨眼
                left_eye_status, left_blink_time, left_is_mouse_down = check_eye_blink("left", left_status,
                                                                                       left_eye_status, left_blink_time,
                                                                                       left_is_mouse_down)
                # 检查右眼是否有意义的眨眼
                right_eye_status, right_blink_time, right_is_mouse_down = check_eye_blink("right", right_status,
                                                                                          right_eye_status,
                                                                                          right_blink_time,
                                                                                          right_is_mouse_down)
        else:
            left_status = 'Unknown'
            right_status = 'Unknown'

        # 显示预测的眼睛状态
        # cv2.putText(frame, f"Left Eye: {leftStatus}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(frame, f"Right Eye: {rightStatus}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # 按下“q”键，退出循环
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
