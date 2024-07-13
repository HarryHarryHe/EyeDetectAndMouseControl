import cv2
import dlib
import numpy as np

# Load the Haar cascade file for eye detection
eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')

# Initialize dlib's face detector and the shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    eyes = []
    for face in faces:
        landmarks = predictor(gray, face)
        # Left eye coordinates
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        # Right eye coordinates
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        eyes.append(left_eye)
        eyes.append(right_eye)
    return eyes

def detect_pupil(eye_roi):
    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=30)

    if circles is not None:
        circles = np.uint8(np.around(circles))
        center = (circles[0][0][0], circles[0][0][1])
        return center
    return None

def draw_cross(frame, center):
    x, y = center
    height, width, _ = frame.shape
    color = (0, 255, 0)  # Green color for the cross
    thickness = 2  # Thickness of the cross lines

    # Draw horizontal line
    cv2.line(frame, (0, y), (width, y), color, thickness)
    # Draw vertical line
    cv2.line(frame, (x, 0), (x, height), color, thickness)
    # Put the text at the center
    cv2.putText(frame, f'({x}, {y})', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect eyes using Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

        eyes = []
        for (ex, ey, ew, eh) in detected_eyes:
            eye_roi = frame[ey:ey+eh, ex:ex+ew]
            eye_center = (ex + ew // 2, ey + eh // 2)
            eyes.append((eye_roi, eye_center))

        for eye_roi, eye_center in eyes:
            center = detect_pupil(eye_roi)
            if center:
                pupil_center = (center[0] + eye_center[0] - eye_roi.shape[1] // 2,
                                center[1] + eye_center[1] - eye_roi.shape[0] // 2)
                draw_cross(frame, pupil_center)

        cv2.imshow('Pupil Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
