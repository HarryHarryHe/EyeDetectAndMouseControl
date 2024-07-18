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

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-m", "--model", required=True, help="path to eye state prediction model")
args = vars(ap.parse_args())

# Initialize dlib's face detector (HOG-based) and facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
model = load_model(args["model"])

# Start the video stream from webcam
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
i = 1
padding = 10  # Set padding pixels to get more eye area for model prediction

# Initialize variables for detecting eye state
left_eye_status = "open"  # Initial state set to open
right_eye_status = "open"
left_blink_time = 0
right_blink_time = 0
blink_threshold = 0.5  # Set time threshold for blink recognition (seconds)
both_eyes_blinked = False  # Flag variable to record if both eyes blinked simultaneously
# Add in initialization area
left_holding = False
right_holding = False


# Load the model
def load_the_model(model_path):
    # models/efficientnetb0-EyeDetection-92.83.h5
    return load_model(model_path)  # Load the model


# Model prediction
def predict_eye_status(eyes, model, eye_image):
    # Expand grayscale image to three channels
    gray_image = np.stack((eye_image,) * 3, axis=-1)

    # Resize image to match model's input size
    gray_image = cv2.resize(gray_image, (224, 224))

    # Add batch dimension for prediction
    gray_image = np.expand_dims(gray_image, axis=0)
    # Use model to predict eye state
    predictions = model.predict(gray_image, verbose=0)  # verbose=0 to not display progress bar
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class = ['open' if predicted_class_index[0] == 1 else 'closed']
    # Print eye state
    # print(f"Predicted {eyes} eye status: {predicted_class}")
    return predicted_class[0]


# Check if eye is blinking
def check_eye_blink(eye, status, previous_status, previous_time, is_holding):
    current_time = time.time()
    # Eye state changes from open to closed, start timing
    if status == "closed" and previous_status == "open":
        return "closed", current_time, is_holding

    # Eye state changes from closed to open, check closure duration
    elif status == "open" and previous_status == "closed":
        # Check closure duration to decide click or hold
        if current_time - previous_time < 0.37:  # Assume less than 0.37 seconds is a click, "0.37 is most suitable for my personal operation habits after testing"
            if not is_holding:
                if eye == "left":
                    pyautogui.click(button='left')  # Perform left click
                    print("Left eye clicked, click left button")
                elif eye == "right":
                    pyautogui.click(button='right')
                    print("Right eye clicked, click right button")
        elif is_holding:
            if eye == "left":
                pyautogui.mouseUp(button='left')  # Release left mouse button
                print("Left eye open, release left button")
            elif eye == "right":
                pyautogui.mouseUp(button='right')  # Release right mouse button
                print("Right eye open, release right button")
        return "open", current_time, False

    # If eye continues to be closed and is already holding, do nothing
    elif status == "closed" and previous_status == "closed" and is_holding:
        # Eye remains closed, but don't press mouse again, only update time
        return "closed", previous_time, is_holding

    # Eye remains closed but hasn't triggered hold yet (exceeds a threshold, e.g. 0.5 seconds)
    elif status == "closed" and previous_status == "closed" and not is_holding:
        if current_time - previous_time > 0.5:
            is_holding = True
            if eye == "left":
                pyautogui.mouseDown(button='left')  # Simulate left mouse button press
                print("Left eye closed, hold left button")
            elif eye == "right":
                pyautogui.mouseDown(button='right')  # Simulate right mouse button press
                print("Right eye closed, hold right button")
        return "closed", previous_time, is_holding

    # State unchanged, return original state
    return previous_status, previous_time, is_holding


# Loop over frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the extended eye area
        leftEyeBounds = [max(min(leftEye[:, 0]) - padding, 0),
                         min(max(leftEye[:, 0]) + padding, gray.shape[1]),
                         max(min(leftEye[:, 1]) - padding, 0),
                         min(max(leftEye[:, 1]) + padding, gray.shape[0])]
        rightEyeBounds = [max(min(rightEye[:, 0]) - padding, 0),
                          min(max(rightEye[:, 0]) + padding, gray.shape[1]),
                          max(min(rightEye[:, 1]) - padding, 0),
                          min(max(rightEye[:, 1]) + padding, gray.shape[0])]

        # Crop eye regions based on extended coordinates
        leftEyeImg = gray[leftEyeBounds[2]:leftEyeBounds[3], leftEyeBounds[0]:leftEyeBounds[1]]
        rightEyeImg = gray[rightEyeBounds[2]:rightEyeBounds[3], rightEyeBounds[0]:rightEyeBounds[1]]

        # Ensure eye region images are not empty
        if leftEyeImg.size > 0 and rightEyeImg.size > 0:
            # Use model to predict eye state
            left_status = predict_eye_status("Left", model, leftEyeImg)
            right_status = predict_eye_status("Right", model, rightEyeImg)

            if left_status == 'closed' and right_status == 'closed':
                both_eyes_blinked = True
                print("Both eyes blinked simultaneously")
            else:
                both_eyes_blinked = False

            if not both_eyes_blinked:
                # Check if left eye has meaningful blink, update in loop
                left_eye_status, left_blink_time, left_holding = check_eye_blink("left", left_status, left_eye_status,
                                                                                 left_blink_time, left_holding)
                # Check if right eye has meaningful blink
                right_eye_status, right_blink_time, right_holding = check_eye_blink("right", right_status,
                                                                                    right_eye_status, right_blink_time,
                                                                                    right_holding)
        else:
            left_status = 'Unknown'
            right_status = 'Unknown'

        # Display predicted eye states
        # cv2.putText(frame, f"Left Eye: {leftStatus}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(frame, f"Right Eye: {rightStatus}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()