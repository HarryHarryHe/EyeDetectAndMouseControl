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
print(screen_width, screen_height)

# 存储上一次手指位置
last_x, last_y = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 将BGR图像转换为RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理图像，检测手部
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 获取食指尖端的坐标
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(tip.x * frame.shape[1]), int(tip.y * frame.shape[0])

            if last_x is not None and last_y is not None:
                # 计算移动距离
                dx, dy = x - last_x, y - last_y

                # 根据距离屏幕边缘的距离调整移动速度
                edge_distance = min(x, screen_width - x, y, screen_height - y)
                speed_factor = max(0.5, min(1.5, edge_distance / 200)) * 2

                # 更新鼠标位置
                pyautogui.moveRel(dx * speed_factor, dy * speed_factor)

            # 更新最后位置
            last_x, last_y = x, y

            # 绘制手部关键点和连线
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # 如果没有检测到手，重置最后位置
        last_x, last_y = None, None

    # 显示图像
    cv2.imshow('MediaPipe Hands', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
