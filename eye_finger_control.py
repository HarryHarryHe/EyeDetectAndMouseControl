import argparse
import time
import dlib
import cv2
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream
from keras.models import load_model
import pyautogui
import mediapipe as mp
import pyttsx3

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

# Initialize pyttsx3 engine for text-to-speech output
engine = pyttsx3.init()
BOTH_EYES_CLOSED_START_TIME = 0
EXIT_BLINK_TIME = 3.0  # 3 seconds of both eyes closed to exit

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


def setup_video_stream():
    """
        Set up and start the video stream from the webcam.

        Returns:
            VideoStream: The initialized video stream object.
    """
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    return vs


def process_frame(frame):
    """
        Process the input frame by flipping, resizing, and converting color spaces.

        Args:
            frame (numpy.ndarray): The input frame from the video stream.

        Returns:
            tuple: A tuple containing the processed frame, grayscale frame, and RGB frame.
    """
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, gray, rgb


def draw_finger_control_area(frame):
    """
        Draw the finger control area on the frame(red point and green rectangle).

        Args:
            frame (numpy.ndarray): The input frame to draw on.

        Returns:
            int: The y-coordinate of the new center of the control area.
    """
    # Calculate the new center for the control(mapping) area
    new_center_y = CONTROL_AREA_HEIGHT // 2 + RECTANGLE_PADDING_TOP
    # Draw a circle at the center of the control(mapping) area
    cv2.circle(frame, (center_x, new_center_y), 10, (0, 0, 255), -1)
    # Draw a rectangle to represent the control(mapping) area
    cv2.rectangle(frame, (center_x - CONTROL_AREA_WIDTH, new_center_y - CONTROL_AREA_HEIGHT),
                  (center_x + CONTROL_AREA_WIDTH, new_center_y + CONTROL_AREA_HEIGHT), (0, 255, 0), 2)
    return new_center_y


def process_hand_landmarks(results, frame, new_center_y):
    """
        Process hand landmarks for mouse control.

        Args:
            results: MediaPipe hand detection results.
            frame (numpy.ndarray): The current video frame.
            new_center_y (int): The y-coordinate of the control area center.
    """
    global CONTROL_ACTIVE, last_x, last_y
    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates of the index finger tip
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(tip.x * frame_width), int(tip.y * frame_height)

            # Check if the finger tip is within the control(mapping) area
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

def extract_eyes_img(rect, gray):
    """
        Extract images of left and right eyes from a face detected in a grayscale image.

        Parameters:
        rect (dlib.rectangle): The rectangle representing the detected face.
        gray (numpy.ndarray): The grayscale image containing the face.

        Returns:
        tuple:
            - leftEyeImg: Grayscale image of the left eye.
            - rightEyeImg: Grayscale image of the right eye.
    """
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Get indices for left and right eyes
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye, rightEye = shape[lStart:lEnd], shape[rStart:rEnd]

    # Ensure left eye is on the left side of the face
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

    return leftEyeImg, rightEyeImg


def process_face_landmarks(rects, gray):
    """
        Process facial landmarks for eye blink detection and mouse control.

        Args:
            rects: Detected face rectangles.
            gray (numpy.ndarray): Grayscale version of the current frame.
    """
    global LEFT_EYE_STATUS, RIGHT_EYE_STATUS, LEFT_BLINK_TIME, RIGHT_BLINK_TIME, LEFT_HOLDING, RIGHT_HOLDING, BOTH_EYES_BLINKED, BOTH_EYES_CLOSED_START_TIME
    # Process each detected face
    for rect in rects:
        # Extract eye images from the face
        leftEyeImg, rightEyeImg = extract_eyes_img(rect, gray)

        # If both eye images are valid
        if leftEyeImg.size > 0 and rightEyeImg.size > 0:
            # Predict eye status (open/closed)
            left_status = predict_eye_status(leftEyeImg)
            right_status = predict_eye_status(rightEyeImg)

            # Check for simultaneous blink
            if left_status == 'closed' and right_status == 'closed':
                if BOTH_EYES_CLOSED_START_TIME == 0:
                    # Initialize the timer for both eyes closed
                    BOTH_EYES_CLOSED_START_TIME = time.time()
                # If both eyes have been closed for 5 seconds, exit the program
                elif time.time() - BOTH_EYES_CLOSED_START_TIME > EXIT_BLINK_TIME:
                    return True  # Signal to exit the program
                BOTH_EYES_BLINKED = True
                # print("Both eyes blinked simultaneously")
            else:
                BOTH_EYES_BLINKED = False
                BOTH_EYES_CLOSED_START_TIME = 0

            # Process individual eye blinks if not a simultaneous blink
            if not BOTH_EYES_BLINKED:
                LEFT_EYE_STATUS, LEFT_BLINK_TIME, LEFT_HOLDING = check_eye_blink("left", left_status, LEFT_EYE_STATUS,
                                                                                 LEFT_BLINK_TIME, LEFT_HOLDING)
                RIGHT_EYE_STATUS, RIGHT_BLINK_TIME, RIGHT_HOLDING = check_eye_blink("right", right_status,
                                                                                    RIGHT_EYE_STATUS, RIGHT_BLINK_TIME,
                                                                                    RIGHT_HOLDING)
        else:
            # Set eye status to unknown if eye images are invalid
            LEFT_EYE_STATUS, RIGHT_EYE_STATUS = 'Unknown', 'Unknown'
    return False


def predict_eye_status(eye_image):
    """
        Predict the status of an eye (open or closed) using a pre-trained model.

        Args:
            eye_image (numpy.ndarray): Grayscale image of an eye.

        Returns:
            str: 'open' if the eye is predicted to be open, 'closed' otherwise.
    """
    gray_image = np.stack((eye_image,) * 3, axis=-1)
    gray_image = cv2.resize(gray_image, (224, 224))
    gray_image = np.expand_dims(gray_image, axis=0)
    predictions = model.predict(gray_image, verbose=0)
    predicted_class_index = np.argmax(predictions, axis=1)
    return 'open' if predicted_class_index[0] == 1 else 'closed'


def check_eye_blink(eye, status, previous_status, previous_time, is_holding):
    """
        Check for eye blinks and handle corresponding mouse actions.

        Args:
            eye (str): Identifier for the eye ('left' or 'right').
            status (str): Current eye status ('open' or 'closed').
            previous_status (str): Previous eye status.
            previous_time (float): Timestamp of the previous blink.
            is_holding (bool): Flag indicating if a mouse button is being held.

        Returns:
            tuple: Updated eye status, blink time, and holding flag.
    """
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
                    speak("Left Click")
                    # print("Left eye clicked, click left button")
                elif eye == "right":
                    pyautogui.click(button='right')  # Perform a single right click operation
                    speak("Right Click")
                    # print("Right eye clicked, click right button")
        # If the eye closure lasts for more than 0.37 seconds, hold the mouse button
        elif is_holding:
            if eye == "left":
                pyautogui.mouseUp(button='left')  # Release left button
                speak("Left Up")
                # print("Left eye open, release left button")
            elif eye == "right":
                pyautogui.mouseUp(button='right')  # Release right button
                speak("Right UP")
                # print("Right eye open, release right button")
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
                speak("Left Down")
                # print("Left eye closed, hold left button")
            elif eye == "right":
                pyautogui.mouseDown(button='right')  # Simulate pressing down the right mouse button
                speak("Right Down")
                # print("Right eye closed, hold right button")
        return "closed", previous_time, is_holding

    # No state change, return the original state
    return previous_status, previous_time, is_holding


def speak(text):
    """
        Speak the given text using the pyttsx3 engine.

        Args:
            text (str): The text to be spoken.
    """
    engine.say(text)
    engine.runAndWait()


def main():
    """
       Main function to run the eye_finger-controlled mouse program.
   """
    vs = setup_video_stream()

    while True:
        frame = vs.read()
        if frame is None:
            print("Failed to read frame from video stream.")
            break

        frame, gray, rgb = process_frame(frame)
        new_center_y = draw_finger_control_area(frame)
        rects = detector(gray, 0)
        results = hands.process(rgb)
        # Add text to the frame indicating "Close eyes 3s to Exit"
        cv2.putText(frame, "Close eyes 3s to Exit", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        process_hand_landmarks(results, frame, new_center_y)
        exit_program = process_face_landmarks(rects, gray)

        if exit_program:
            print("Exiting program due to long eye closure")
            speak("System Exit")
            break

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            speak("System Exit")
            break

    cv2.destroyAllWindows()
    vs.stop()


def wait_for_blink():
    """
    Wait for the user to blink before starting the main program.

    Returns: None
    """
    print("Waiting for blink to start the program...")

    vs = VideoStream(src=0).start()
    time.sleep(3.0)
    speak("Blink to start the program")

    while True:
        frame = vs.read()
        if frame is None:
            continue

        frame, gray, _ = process_frame(frame)
        rects = detector(gray, 0)

        # Add text to the frame indicating "Blink to Link"
        cv2.putText(frame, "Blink to Link", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for rect in rects:
            leftEye, rightEye = extract_eyes_img(rect, gray)

            if leftEye.size > 0 and rightEye.size > 0:
                leftEyeStatus = predict_eye_status(leftEye)
                rightEyeStatus = predict_eye_status(rightEye)

                if leftEyeStatus == 'closed' and rightEyeStatus == 'closed':
                    vs.stop()
                    cv2.destroyAllWindows()
                    return

        cv2.imshow("Blink to Start", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    wait_for_blink()
    speak("System Starting")
    main()
