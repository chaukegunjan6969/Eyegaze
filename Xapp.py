from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import torch
import numpy as np
from PIL import Image
import io
from models import gazenet
from mtcnn import FaceDetector
import utils

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load model and configure device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = gazenet.GazeNet(device)
model.load_state_dict(torch.load('models/weights/gazenet.pth', map_location=device))
model.eval()
face_detector = FaceDetector(device=device)

# OpenCV video capture
cap = cv2.VideoCapture(0)

def generate_video_feed():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(frame_rgb)
        screen_center = (img_w // 2, img_h // 2)
        
        faces, landmarks = face_detector.detect(Image.fromarray(frame_rgb))

        if faces:
            for f, lm in zip(faces, landmarks):
                if f[-1] > 0.98:  # Confidence threshold
                    face, gaze_origin, M = utils.normalize_face(lm, frame_rgb)
                    with torch.no_grad():
                        gaze = model.get_gaze(face)
                        gaze = gaze[0].data.cpu()
                    gaze_x = int(gaze_origin[0] + gaze[0] * 100)
                    gaze_y = int(gaze_origin[1] - gaze[1] * 100)
                    relative_coords = (gaze_x - screen_center[0], gaze_y - screen_center[1])

                    # Draw gaze direction and coordinates on frame
                    frame = cv2.putText(frame, f'Coords: {relative_coords}', 
                                        (gaze_origin[0] + 10, gaze_origin[1] - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    frame = utils.draw_gaze(frame, gaze_origin, gaze, color=(255, 0, 0), thickness=2)

        # Encode frame to JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_data = jpeg.tobytes()
        
        # Yield frame data in HTTP response format for video streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
