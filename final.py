import os
import sys
import cv2
import time
import torch
import utils
import argparse
import traceback
import numpy as np

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

                    # Calculate projected gaze endpoint based on formula
                    t = avg_eye_center[2] / avg_gaze[2]  # depth factor
                    gaze_x = avg_eye_center[0] + t * avg_gaze[0]
                    gaze_y = avg_eye_center[1] + t * avg_gaze[1]

                    # Convert from meters to pixels, rounding off
                    pixel_x = int(round(gaze_x))
                    pixel_y = int(round(gaze_y))

                    # Display coordinates relative to screen center
                    relative_coords = (pixel_x - screen_center[0], pixel_y - screen_center[1])
                    display = cv2.putText(display, f'Coords: {relative_coords}', 
                                          (gaze_origin[0] + 10, gaze_origin[1] - 10), 
                                          cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2) 

                    # Display warning if gaze is outside the screen bounds
                    if pixel_x < 0 or pixel_x > img_w or pixel_y < 0 or pixel_y > img_h:
                        display = cv2.putText(display, 'Warning: Looking outside screen', 
                                              (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

                    # Draw gaze point and direction
                    display = cv2.circle(display, (int(gaze_origin[0]), int(gaze_origin[1])), 3, (0, 255, 0), -1)
                    display = utils.draw_gaze(display, gaze_origin, avg_gaze, color=(255, 0, 0), thickness=2)

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
