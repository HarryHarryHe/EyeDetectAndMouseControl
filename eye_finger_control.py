import argparse
import time
import dlib
import cv2
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream
import keras
from keras.models import load_model
import pyautogui
import mediapipe as mp

# Initialize dlib's face detector and facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
model = load_model('models/efficientnetb0-EyeDetection-92.83.h5')

# Initialize MediaPipe hand gesture recognition
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start video stream from webcam
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Constants for eye extraction and control area
EYE_EXTRACT_PADDING = 10  # Padding for eye region extraction
RECTANGLE_PADDING_TOP = 50  # Top padding for control area rectangle

BLINK_THRESHOLD_TIME = 0.37  # Time threshold for blink detection
BLINK_CLOSED_TIME = 0.5  # Time threshold for mouse long press detection

# Set the number of pixels to expand the boundary of the control area for easy mouse edge control and mapping
CONTROL_AREA_WIDTH_PADDING, CONTROL_AREA_HEIGHT_PADDING = 75, 75

CONTROL_AREA_WIDTH, CONTROL_AREA_HEIGHT = 160, 80  # Size of control area

CONTROL_ACTIVE = False  # Flag for mouse control activation

# Initialize variables for eye state tracking
LEFT_EYE_STATUS, RIGHT_EYE_STATUS = "open", "open"
LEFT_BLINK_TIME, RIGHT_BLINK_TIME = 0, 0
BOTH_EYES_BLINKED, LEFT_HOLDING, RIGHT_HOLDING = False, False, False

# Get screen size
screen_width, screen_height = pyautogui.size()
print("Screen size: ", screen_width, screen_height)

# Get video frame size
frame = vs.read()
if frame is None:
    print("Failed to read frame from video stream.")
    vs.stop()
    cv2.destroyAllWindows()
    exit()
frame_height, frame_width, _ = frame.shape

# Calculate center of the video frame
center_x, center_y = frame_width // 2, frame_height // 2

# Initialize smooth filter
last_x, last_y = screen_width // 2, screen_height // 2
SMOOTH_FACTOR = 3  # Smoothing factor for mouse movement

'''
    @description: Function to predict eye status (open/closed)
    @param: eye_image: eye image for prediction
    @return: predicted eye status (open/closed)
'''
def predict_eye_status(eye_image):
    gray_image = np.stack((eye_image,) * 3, axis=-1)
    gray_image = cv2.resize(gray_image, (224, 224))
    gray_image = np.expand_dims(gray_image, axis=0)
    predictions = model.predict(gray_image, verbose=0)
    predicted_class_index = np.argmax(predictions, axis=1)
    return 'open' if predicted_class_index[0] == 1 else 'closed'


'''
    @description: Function to check for eye blinks and handle mouse actions
    @param: eye: left or right eye
    @param: status: current eye status
    @param: previous_status: previous eye status
    @param: previous_time: previous blink time
    @param: is_holding: flag for holding mouse button
    @return: updated eye status, blink time, and holding flag
'''
def check_eye_blink(eye, status, previous_status, previous_time, is_holding):
    current_time = time.time()
    # Eye state transitions and corresponding actions
    if status == "closed" and previous_status == "open":
        return "closed", current_time, is_holding

    # Check the duration of the eye closure when the eye state changes from closed to open
    elif status == "open" and previous_status == "closed":
        # Check the duration of the closure to decide whether to click or hold
        # Assuming less than 0.37 seconds is a click, ""more than 0.37 is a long press,
        # tested 0.37 is most suitable for my personal habits ""
        if current_time - previous_time < BLINK_THRESHOLD_TIME:
            if not is_holding:
                if eye == "left":
                    pyautogui.click(button='left')  # Perform a single left click operation
                    print("Left eye clicked, click left button")
                elif eye == "right":
                    pyautogui.click(button='right')  # Perform a single right click operation
                    print("Right eye clicked, click right button")
        # If the eye closure lasts for more than 0.37 seconds, hold the mouse button
        elif is_holding:
            if eye == "left":
                pyautogui.mouseUp(button='left')  # Release left button
                print("Left eye open, release left button")
            elif eye == "right":
                pyautogui.mouseUp(button='right')  # Release right button
                print("Right eye open, release right button")
        return "open", current_time, False

    # If the eye continues to be closed and has already been held down, do nothing
    elif status == "closed" and previous_status == "closed" and is_holding:
        # The eye continues to be closed, but do not press the mouse again, only update the time
        return "closed", previous_time, is_holding

    # The eye continues to be closed but has not yet triggered a long press (exceeding a certain threshold, such as 0.5 seconds)
    elif status == "closed" and previous_status == "closed" and not is_holding:
        if current_time - previous_time > BLINK_CLOSED_TIME:
            is_holding = True
            if eye == "left":
                pyautogui.mouseDown(button='left')  # Simulate pressing down the left mouse button
                print("Left eye closed, hold left button")
            elif eye == "right":
                pyautogui.mouseDown(button='right')  # Simulate pressing down the right mouse button
                print("Right eye closed, hold right button")
        return "closed", previous_time, is_holding

    # No state change, return the original state
    return previous_status, previous_time, is_holding


# Main loop for processing video frames
while True:
    # Read a frame from the video stream
    frame = vs.read()
    if frame is None:
        print("Failed to read frame from video stream.")
        vs.stop()
        cv2.destroyAllWindows()
        exit()

    # Flip the frame horizontally and resize it，convenient image processing
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame)
    # Convert the frame to grayscale and RGB
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Calculate the new center for the control area
    new_center_y = CONTROL_AREA_HEIGHT // 2 + RECTANGLE_PADDING_TOP
    # Draw a circle at the center of the control area
    cv2.circle(frame, (center_x, new_center_y), 10, (0, 0, 255), -1)
    # Draw a rectangle to represent the control(mapping) area
    cv2.rectangle(frame, (center_x - CONTROL_AREA_WIDTH, new_center_y - CONTROL_AREA_HEIGHT),
                  (center_x + CONTROL_AREA_WIDTH, new_center_y + CONTROL_AREA_HEIGHT), (0, 255, 0), 2)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    # Process the RGB frame for hand detection
    results = hands.process(rgb)

    # If hand landmarks are detected, check if the tip of the index finger is within the control area
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates of the index finger tip
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(tip.x * frame_width), int(tip.y * frame_height)
            # Check if the finger tip is within the control area
            if (center_x - CONTROL_AREA_WIDTH <= x <= center_x + CONTROL_AREA_WIDTH) and (
                    new_center_y - CONTROL_AREA_HEIGHT <= y <= new_center_y + CONTROL_AREA_HEIGHT):
                # Activate mouse control if not already active
                if not CONTROL_ACTIVE:
                    CONTROL_ACTIVE = True
                    pyautogui.moveTo(screen_width / 2, screen_height / 2)

                # Map finger position to screen coordinates
                screen_x = np.interp(x, [center_x - CONTROL_AREA_WIDTH, center_x + CONTROL_AREA_WIDTH],
                                     [0 - CONTROL_AREA_WIDTH_PADDING, screen_width + CONTROL_AREA_WIDTH_PADDING])
                screen_y = np.interp(y, [new_center_y - CONTROL_AREA_HEIGHT, new_center_y + CONTROL_AREA_HEIGHT],
                                     [0 - CONTROL_AREA_HEIGHT_PADDING, screen_height + CONTROL_AREA_HEIGHT_PADDING])

                # Apply smoothing to the mouse movement
                screen_x = (last_x * (SMOOTH_FACTOR - 1) + screen_x) / SMOOTH_FACTOR
                screen_y = (last_y * (SMOOTH_FACTOR - 1) + screen_y) / SMOOTH_FACTOR
                pyautogui.moveTo(screen_x, screen_y)

                # Update last position for smoothing
                last_x, last_y = screen_x, screen_y

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Process each detected face
    for rect in rects:
        # Predict facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Get indices for left and right eyes
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye, rightEye = shape[lStart:lEnd], shape[rStart:rEnd]

        # Ensure left eye is on the left side of the face and right eye is on the right side of the face
        if leftEye[:, 0].mean() > rightEye[:, 0].mean():
            leftEye, rightEye = rightEye, leftEye

        # Calculate boundaries for eye regions with padding
        leftEyeBounds = [max(min(leftEye[:, 0]) - EYE_EXTRACT_PADDING, 0),
                         min(max(leftEye[:, 0]) + EYE_EXTRACT_PADDING, gray.shape[1]),
                         max(min(leftEye[:, 1]) - EYE_EXTRACT_PADDING, 0),
                         min(max(leftEye[:, 1]) + EYE_EXTRACT_PADDING, gray.shape[0])]
        rightEyeBounds = [max(min(rightEye[:, 0]) - EYE_EXTRACT_PADDING, 0),
                          min(max(rightEye[:, 0]) + EYE_EXTRACT_PADDING, gray.shape[1]),
                          max(min(rightEye[:, 1]) - EYE_EXTRACT_PADDING, 0),
                          min(max(rightEye[:, 1]) + EYE_EXTRACT_PADDING, gray.shape[0])]

        # Extract eye images
        leftEyeImg = gray[leftEyeBounds[2]:leftEyeBounds[3], leftEyeBounds[0]:leftEyeBounds[1]]
        rightEyeImg = gray[rightEyeBounds[2]:rightEyeBounds[3], rightEyeBounds[0]:rightEyeBounds[1]]

        # If both eye images are valid
        if leftEyeImg.size > 0 and rightEyeImg.size > 0:
            # Predict eye status (open/closed)
            left_status = predict_eye_status(leftEyeImg)
            right_status = predict_eye_status(rightEyeImg)

            # Check for simultaneous blink
            if left_status == 'closed' and right_status == 'closed':
                BOTH_EYES_BLINKED = True
                print("Both eyes blinked simultaneously")
            else:
                BOTH_EYES_BLINKED = False

            # Process individual eye blinks if not a simultaneous blink
            if not BOTH_EYES_BLINKED:
                LEFT_EYE_STATUS, LEFT_BLINK_TIME, LEFT_HOLDING = check_eye_blink("left", left_status, LEFT_EYE_STATUS,
                                                                                 LEFT_BLINK_TIME, LEFT_HOLDING)
                RIGHT_EYE_STATUS, RIGHT_BLINK_TIME, RIGHT_HOLDING = check_eye_blink("right", right_status,
                                                                                    RIGHT_EYE_STATUS, RIGHT_BLINK_TIME,
                                                                                    RIGHT_HOLDING)
        else:
            # Set eye status to unknown if eye images are invalid
            left_status, right_status = 'Unknown', 'Unknown'

    # Display the processed frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' is pressed
    if key == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()
