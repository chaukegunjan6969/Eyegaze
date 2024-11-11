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
                        gaze = gaze[0].data.cpu()

                    # Calculate the gaze endpoint coordinates
                    gaze_x = int(gaze_origin[0] + gaze[0] * 100)  # Scale factor for visualization
                    gaze_y = int(gaze_origin[1] - gaze[1] * 100)  # Flip y-axis for display

                    # Display coordinates relative to screen center
                    relative_coords = (gaze_x - screen_center[0], gaze_y - screen_center[1])
                    display = cv2.putText(display, f'Coords: {relative_coords}', 
                                          (gaze_origin[0] + 10, gaze_origin[1] - 10), 
                                          cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2) 

                    # Display warning if gaze is outside the screen bounds
                    if gaze_x < 0 or gaze_x > img_w or gaze_y < 0 or gaze_y > img_h:
                        display = cv2.putText(display, 'Warning: Looking outside screen', 
                                              (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

                    # Draw gaze point and direction
                    display = cv2.circle(display, gaze_origin, 3, (0, 255, 0), -1)
                    display = utils.draw_gaze(display, gaze_origin, gaze, color=(255, 0, 0), thickness=2)

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
