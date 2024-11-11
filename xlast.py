import os
import sys
import cv2
import time
import torch
import utils
import argparse
import traceback
import numpy as np
import math

from PIL import Image
from models import gazenet
from mtcnn import FaceDetector
from collections import deque

# Argument parser for optional CPU usage and model weights
parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--weights', '-w', type=str, default='models/weights/gazenet.pth')
args = parser.parse_args()

print('Loading MobileFaceGaze model...')
device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.cpu) else "cpu")
model = gazenet.GazeNet(device)

if not torch.cuda.is_available() and not args.cpu:
    print('Tried to load GPU but found none. Please check your environment')

# Load model weights
state_dict = torch.load(args.weights, map_location=device)
model.load_state_dict(state_dict)
print('Model loaded using {} as device'.format(device))

model.eval()

fps = 0
frame_num = 0
frame_samples = 6
fps_timer = time.time()
cap = cv2.VideoCapture(0)

face_detector = FaceDetector(device=device)

# Variables to store gaze data for averaging
eye_centers = deque(maxlen=3)
gaze_vectors = deque(maxlen=3)

# Define pitch and yaw thresholds based on observed values
PITCH_UP = 12     # Looking too high (adjust based on observed values)
PITCH_DOWN = -15   # Looking too low
YAW_LEFT = -15     # Looking too far left
YAW_RIGHT = 15     # Looking too far right
warning_count = 0  # Initialize warning counter

while True:
    try:
        ret, frame = cap.read()
        frame = frame[:, :, ::-1]  # Convert BGR to RGB
        frame = cv2.flip(frame, 1)  # Flip for mirror effect
        img_h, img_w, _ = np.shape(frame)
        frame_num += 1
        display = frame.copy()

        # Center of the laptop screen (considered as origin)
        screen_center = (img_w // 2, img_h // 2)

        # Detect faces and landmarks
        faces, landmarks = face_detector.detect(Image.fromarray(frame))

        if len(faces) != 0:
            for f, lm in zip(faces, landmarks):
                if f[-1] > 0.98:  # Confidence threshold
                    face, gaze_origin, M = utils.normalize_face(lm, frame)

                    # Predict gaze direction
                    with torch.no_grad():
                        gaze = model.get_gaze(face)
                        gaze = gaze[0].data.cpu().numpy()  # Convert to numpy array

                    # Ensure gaze has a third element (z) for calculation
                    gaze = np.append(gaze, 1) if gaze.size == 2 else gaze

                    # Store eye center and gaze vector
                    eye_center = (gaze_origin[0], gaze_origin[1], 1)  # Assuming a depth of 1 for simplicity
                    eye_centers.append(eye_center)
                    gaze_vectors.append(gaze)

                    # Calculate averaged gaze vector over last 3 frames
                    avg_gaze = np.mean(gaze_vectors, axis=0) if len(gaze_vectors) == 3 else gaze
                    avg_eye_center = np.mean(eye_centers, axis=0) if len(eye_centers) == 3 else eye_center

                    # Calculate pitch and yaw angles
                    x, y, z = avg_gaze
                    pitch = math.degrees(math.atan2(y, math.sqrt(x*2 + z*2)))
                    yaw = math.degrees(math.atan2(x, z))

                    # Check if gaze is outside screen bounds
                    if pitch > PITCH_UP or pitch < PITCH_DOWN or yaw < YAW_LEFT or yaw > YAW_RIGHT:
                        warning_count += 1
                        print(f'Warning {warning_count}/3: Gaze outside screen bounds')

                        if warning_count <= 3:
                            # Display warning message if within the 3-warning limit
                            display = cv2.putText(display, f'Warning {warning_count}: Look Inside Screen',
                                                  (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                        else:
                            warning_count = 0  # Reset count after 3 warnings

                    # Print pitch and yaw in the terminal
                    print(f'Pitch: {pitch:.2f}, Yaw: {yaw:.2f}')

        # Calculate FPS
        if frame_num == frame_samples:
            fps = time.time() - fps_timer
            fps = frame_samples / fps
            fps_timer = time.time()
            frame_num = 0
        display = cv2.putText(display, 'FPS: {:.2f}'.format(fps), (0, 20), 
                              cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)

        # Show the display window
        cv2.imshow('Gaze Demo', cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break

    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        cap.release()
        cv2.destroyAllWindows()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
        break

cap.release()
cv2.destroyAllWindows()
