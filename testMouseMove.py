import cv2
import numpy as np
import pyautogui
from keras.models import load_model

# 载入模型
model_path = "models/conv2d_6-EyeMouseMapper.h5"
model = load_model(model_path)

# 设置网络摄像头
cap = cv2.VideoCapture(0)  # 0 代表默认摄像头

# 载入眼部检测器
eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')

# 设置屏幕分辨率
screen_width, screen_height = pyautogui.size()
i = 1
while True:
    # 捕获摄像头中的图像
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测眼睛
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    for (ex, ey, ew, eh) in eyes:
        # 提取眼部区域
        eye_img = frame[ey:ey + eh, ex:ex + ew]
        eye_img = cv2.resize(eye_img, (448, 224))  # 调整大小以匹配模型输入
        if i==1:
            cv2.imwrite('eye.jpg', eye_img)
            i=0
        eye_img = eye_img / 255.0  # 归一化
        eye_img = np.expand_dims(eye_img, axis=0)  # 增加批次维度

        # 预测鼠标位置
        pred = model.predict(eye_img,verbose=0)
        print("Predicted position:",pred[0][0],pred[0][1])
        x_pred, y_pred = pred[0][0] * screen_width, pred[0][1] * screen_height
        print("Mapped position:",x_pred, y_pred)
        # 移动鼠标
        pyautogui.moveTo(x_pred, y_pred,1)
        break  # 假设只处理第一只检测到的眼睛

    # 显示结果
    cv2.imshow('Webcam', frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
