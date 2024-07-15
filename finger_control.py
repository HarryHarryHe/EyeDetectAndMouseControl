import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# 初始化MediaPipe手势识别
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

# 初始化MediaPipe绘图工具
mp_drawing = mp.solutions.drawing_utils

# 捕获视频
cap = cv2.VideoCapture(0)

# 获取屏幕尺寸
screen_width, screen_height = pyautogui.size()

# 获取视频帧尺寸
ret, frame = cap.read()
frame_height, frame_width, _ = frame.shape

# 计算视频帧中心区域的边界
center_x, center_y = frame_width // 2, frame_height // 2
center_area_width, center_area_height = 128, 64  # 中心监控区域的一半宽度和高度

control_active = False  # 控制鼠标的激活状态

# 平滑滤波器初始化
last_x, last_y = screen_width // 2, screen_height // 2
smooth_factor = 5
while True:
    # 读取视频帧并进行处理
    ret, frame = cap.read()
    if not ret:
        break

    # 在视频帧中心绘制红色圆形标记
    cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)  # 绘制实心圆

    # 绘制中心监控区域的绿色矩形框
    cv2.rectangle(frame, (center_x - center_area_width, center_y - center_area_height),
                  (center_x + center_area_width, center_y + center_area_height), (0, 255, 0), 2)

    # 将BGR图像转换为RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理图像，检测手部
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 获取食指尖端的坐标
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(tip.x * frame_width), int(tip.y * frame_height)

            # 检查手指是否在中心监控区域内
            if (center_x - center_area_width <= x <= center_x + center_area_width) and (
                    center_y - center_area_height <= y <= center_y + center_area_height):
                if not control_active:
                    control_active = True
                    pyautogui.moveTo(screen_width / 2, screen_height / 2)

                # 映射手指的位置到整个屏幕
                screen_x = np.interp(x, [center_x - center_area_width, center_x + center_area_width], [0, screen_width])
                screen_y = np.interp(y, [center_y - center_area_height, center_y + center_area_height],
                                     [0, screen_height])

                # 应用平滑滤波
                screen_x = (last_x * (smooth_factor - 1) + screen_x) / smooth_factor
                screen_y = (last_y * (smooth_factor - 1) + screen_y) / smooth_factor
                pyautogui.moveTo(screen_x, screen_y)

                last_x, last_y = screen_x, screen_y  # 更新上一次的坐标

            # 绘制手部关键点和连线
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 显示图像
    cv2.imshow('MediaPipe Hands', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
