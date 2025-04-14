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

def get_eye_regions(lm):
    if len(lm) < 68:
        return None, None  # Ensure there are enough landmarks
    left_eye = lm[36:42]
    right_eye = lm[42:48]
    return left_eye, right_eye

def calculate_ear(eye):
    if eye is None or len(eye) < 6:
        return None  # Avoid processing invalid data
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--weights','-w', type=str, default='models/weights/gazenet.pth')
args = parser.parse_args()

print('Loading MobileFaceGaze model...')
device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.cpu) else "cpu")
model = gazenet.GazeNet(device)

if(not torch.cuda.is_available() and not args.cpu):
    print('Tried to load GPU but found none. Please check your environment')

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

keys = [['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M']]

screen_w, screen_h = 1280, 720
key_w, key_h = screen_w // 10, screen_h // 3

def get_key_from_coords(x, y):
    col = x // key_w
    row = y // key_h
    if row < len(keys) and col < len(keys[row]):
        return keys[row][col]
    return None

blink_count = 0
blink_threshold = 3
selected_text = ""

while True:
    try:
        ret, frame = cap.read()
        frame = frame[:,:,::-1]
        frame = cv2.flip(frame, 1)
        img_h, img_w, _ = np.shape(frame)
        frame_num += 1
        display = frame.copy()
        faces, landmarks = face_detector.detect(Image.fromarray(frame))
        
        if len(faces) != 0:
            for f, lm in zip(faces, landmarks):
                if(f[-1] > 0.98):
                    face, gaze_origin, M = utils.normalize_face(lm, frame)
                    
                    with torch.no_grad():
                        gaze = model.get_gaze(face)
                        gaze = gaze[0].data.cpu().numpy()
                    
                    x_coord = int(gaze_origin[0] + gaze[0] * 1000 / 100)
                    y_coord = int(gaze_origin[1] + gaze[1] * 1000 / 100)
                    
                    key = get_key_from_coords(x_coord, y_coord)
                    if key:
                        display = cv2.putText(display, f'Key: {key}', (50, 100), 
                                              cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                    
                    left_eye, right_eye = get_eye_regions(lm)
                    if left_eye is not None and right_eye is not None:
                        ear_left = calculate_ear(left_eye)
                        ear_right = calculate_ear(right_eye)
                        if ear_left is not None and ear_right is not None:
                            ear = (ear_left + ear_right) / 2.0
                            if ear < 0.2:
                                blink_count += 1
                            else:
                                if blink_count >= blink_threshold and key:
                                    selected_text += key
                                blink_count = 0
                    
        display = cv2.putText(display, 'Typed: ' + selected_text, (50, 150), 
                              cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        
        if frame_num == frame_samples:
            fps = frame_samples / (time.time() - fps_timer)
            fps_timer = time.time()
            frame_num = 0
        
        display = cv2.putText(display, 'FPS: {:.2f}'.format(fps), (0, 20), 
                              cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
        
        cv2.imshow('Gaze Keyboard', cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
        break

cap.release()
cv2.destroyAllWindows()
