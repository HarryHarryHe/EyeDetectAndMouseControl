import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

# Initialize MediaPipe drawing tools
mp_drawing = mp.solutions.drawing_utils

# Capture video
cap = cv2.VideoCapture(0)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Get video frame dimensions
ret, frame = cap.read()
frame_height, frame_width, _ = frame.shape

# Calculate boundaries of the center area in the video frame
center_x, center_y = frame_width // 2, frame_height // 2
center_area_width, center_area_height = 128, 64  # Half width and height of the central monitoring area

control_active = False  # Mouse control activation status

# Initialize smooth filter
last_x, last_y = screen_width // 2, screen_height // 2
# Smoothing factor
smooth_factor = 5

while True:
    # Read and process video frame
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a red circular marker in the center of the video frame
    cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)  # Draw a solid circle

    # Draw a green rectangular frame for the central monitoring area
    cv2.rectangle(frame, (center_x - center_area_width, center_y - center_area_height),
                  (center_x + center_area_width, center_y + center_area_height), (0, 255, 0), 2)

    # Convert BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get coordinates of the index finger tip
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(tip.x * frame_width), int(tip.y * frame_height)

            # Check if the finger is within the central monitoring area
            if (center_x - center_area_width <= x <= center_x + center_area_width) and (
                    center_y - center_area_height <= y <= center_y + center_area_height):
                if not control_active:
                    control_active = True
                    pyautogui.moveTo(screen_width / 2, screen_height / 2)

                # Map finger position to the entire screen
                screen_x = np.interp(x, [center_x - center_area_width, center_x + center_area_width], [0, screen_width])
                screen_y = np.interp(y, [center_y - center_area_height, center_y + center_area_height], [0, screen_height])

                # Apply smooth filter
                screen_x = (last_x * (smooth_factor - 1) + screen_x) / smooth_factor
                screen_y = (last_y * (smooth_factor - 1) + screen_y) / smooth_factor

                pyautogui.moveTo(screen_x, screen_y)
                last_x, last_y = screen_x, screen_y  # Update last coordinates

            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the image
    cv2.imshow('MediaPipe Hands', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()