import cv2
import numpy as np
import pyautogui
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def calibrate():
    # 简单的校准过程
    calibration_points = [(100, 100), (100, 900), (1820, 100), (1820, 900)]
    calibration_data = []

    for point in calibration_points:
        pyautogui.moveTo(point)
        print(f"请看着屏幕上的点 {point}")
        input("按回车继续...")
        # 这里应该获取当前的眼睛位置
        # 为简化，我们使用随机值代替
        eye_pos = (np.random.randint(0, 1280), np.random.randint(0, 720))
        calibration_data.append((eye_pos, point))

    return calibration_data


def map_eye_to_screen(left_eye, right_eye, calibration_data):
    # 计算两眼的中心点
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # 使用简单的线性插值来映射眼睛位置到屏幕坐标
    # 这里使用了非常简化的方法，实际应用中可能需要更复杂的算法
    x = np.interp(eye_center[0],
                  [min(c[0][0] for c in calibration_data), max(c[0][0] for c in calibration_data)],
                  [min(c[1][0] for c in calibration_data), max(c[1][0] for c in calibration_data)])
    y = np.interp(eye_center[1],
                  [min(c[0][1] for c in calibration_data), max(c[0][1] for c in calibration_data)],
                  [min(c[1][1] for c in calibration_data), max(c[1][1] for c in calibration_data)])

    return int(x), int(y)


def smooth_coordinates(x, y, prev_x, prev_y, alpha=0.3):
    return alpha * x + (1 - alpha) * prev_x, alpha * y + (1 - alpha) * prev_y


# 初始化dlib的人脸检测器和特征预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 校准
print("开始校准过程...")
calibration_data = calibrate()
print("校准完成!")

prev_x, prev_y = pyautogui.position()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # 计算眼睛中心
        leftEyeCenter = leftEye.mean(axis=0).astype("int")
        rightEyeCenter = rightEye.mean(axis=0).astype("int")

        # 使用校准数据将眼睛位置映射到屏幕坐标
        x, y = map_eye_to_screen(leftEyeCenter, rightEyeCenter, calibration_data)

        # 平滑处理
        x, y = smooth_coordinates(x, y, prev_x, prev_y)

        # 移动鼠标
        pyautogui.moveTo(x, y)

        prev_x, prev_y = x, y

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()