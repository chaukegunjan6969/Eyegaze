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
while True:
    try:
        ret, frame = cap.read()
        frame = frame[:,:,::-1]
        frame = cv2.flip(frame, 1)
        img_h, img_w, _ = np.shape(frame)
        frame_num += 1
        # Detect Faces
        display = frame.copy()
        faces, landmarks = face_detector.detect(Image.fromarray(frame))

        # if len(faces) != 0:
        #     for f, lm in zip(faces, landmarks):
        #         # Confidence check
        #         if(f[-1] > 0.98):
        #             # Crop and normalize face Face
        #             face, gaze_origin, M  = utils.normalize_face(lm, frame)
                                        
        #             # Predict gaze
        #             with torch.no_grad():
        #                 gaze = model.get_gaze(face)
        #                 gaze = gaze[0].data.cpu()                              
                    

        #             # Draw results
        #             display = cv2.circle(display, gaze_origin, 3, (0, 255, 0), -1)            
        #             display = utils.draw_gaze(display, gaze_origin, gaze, color=(255,0,0), thickness=2)
        if len(faces) != 0:
            for f, lm in zip(faces, landmarks):
        # Confidence check
             if(f[-1] > 0.98):
            # Crop and normalize face
              face, gaze_origin, M = utils.normalize_face(lm, frame)
                                        
                # Predict gaze
            with torch.no_grad():
                 gaze = model.get_gaze(face)
                 gaze = gaze[0].data.cpu().numpy()  # Convert tensor to numpy array for calculations

            # Project the gaze onto a 2D plane (assuming a simple screen plane at a certain distance)
            # Adjust the scaling factor to match the resolution and size of your display
            screen_distance = 1000  # Distance from camera to screen (in pixels or chosen units)
            scaling_factor = 100   # Adjust based on desired output units
            
            # Calculate X-Y coordinates on the screen
            x_coord = int(gaze_origin[0] + gaze[0] * screen_distance / scaling_factor)
            y_coord = int(gaze_origin[1] + gaze[1] * screen_distance / scaling_factor)
            
                # Draw the coordinates on the display
            display = cv2.putText(display, f'X: {x_coord}, Y: {y_coord}', (50, 50), 
                                      cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

                # Optionally draw a point where the gaze lands
            display = cv2.circle(display, (x_coord, y_coord), 5, (0, 0, 255), -1)


        # Calc FPS
        if (frame_num == frame_samples):
            fps = time.time() - fps_timer
            fps  = frame_samples / fps;
            fps_timer = time.time()
            frame_num = 0
        display = cv2.putText(display, 'FPS: {:.2f}'.format(fps), (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow('Gaze Demo', cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        cap.release()               
        cv2.destroyAllWindows()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
        break

cap.release()
cv2.destroyAllWindows()

