import os
import sys
import cv2
import time
import torch
import utils
import argparse
import traceback
import numpy as np
from collections import deque
from PIL import Image
from models import gazenet
from mtcnn import FaceDetector

class VirtualKeyboard:
    def __init__(self, rows=3, cols=10):
        self.rows = rows
        self.cols = cols
        self.keys = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/']
        ]
        self.key_regions = None
        self.key_size = (60, 60)  # width, height
        self.key_padding = 5
        self.keyboard_pos = (50, 150)  # x, y position of keyboard
        self.text = ""
        self.text_pos = (50, 100)
        self.calibration_points = []
        self.calibration_data = []
        self.calibrated = False
        self.calibration_threshold = 0.1  # Normalized distance threshold
        self.dwell_time_threshold = 1.0  # seconds to select a key
        self.gaze_history = deque(maxlen=10)  # Store recent gaze points
        self.current_key = None
        self.key_selection_start = None
        
    def draw_keyboard(self, frame):
        """Draw the virtual keyboard on the frame"""
        key_w, key_h = self.key_size
        padding = self.key_padding
        start_x, start_y = self.keyboard_pos
        
        self.key_regions = []
        
        # Draw text input area
        cv2.rectangle(frame, 
                      (self.text_pos[0], self.text_pos[1] - 30),
                      (self.text_pos[0] + 600, self.text_pos[1] + 10),
                      (200, 200, 200), -1)
        cv2.putText(frame, self.text, 
                   (self.text_pos[0] + 5, self.text_pos[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Draw keyboard keys
        for row in range(self.rows):
            for col in range(self.cols):
                x1 = start_x + col * (key_w + padding)
                y1 = start_y + row * (key_h + padding)
                x2 = x1 + key_w
                y2 = y1 + key_h
                
                # Store key regions for hit testing
                self.key_regions.append({
                    'key': self.keys[row][col],
                    'rect': (x1, y1, x2, y2),
                    'center': ((x1+x2)//2, (y1+y2)//2)
                })
                
                # Draw key
                color = (255, 255, 255)
                if self.current_key == self.keys[row][col]:
                    color = (0, 255, 0)  # Highlight current key
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                
                # Draw key label
                text_size = cv2.getTextSize(self.keys[row][col], 
                                           cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.8, 2)[0]
                text_x = x1 + (key_w - text_size[0]) // 2
                text_y = y1 + (key_h + text_size[1]) // 2
                cv2.putText(frame, self.keys[row][col], 
                           (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return frame
    
    def check_key_press(self, gaze_point):
        """Check if gaze is on a key and handle selection"""
        if not self.key_regions:
            return
        
        # Find which key is being looked at
        new_key = None
        for key_data in self.key_regions:
            x1, y1, x2, y2 = key_data['rect']
            if x1 <= gaze_point[0] <= x2 and y1 <= gaze_point[1] <= y2:
                new_key = key_data['key']
                break
        
        # Key selection logic
        if new_key != self.current_key:
            self.current_key = new_key
            self.key_selection_start = time.time() if new_key else None
        elif new_key and (time.time() - self.key_selection_start) > self.dwell_time_threshold:
            self.text += new_key
            self.current_key = None
            self.key_selection_start = None
    
    def add_calibration_point(self, point, actual_pos):
        """Add a calibration data point"""
        self.calibration_data.append({
            'gaze_point': point,
            'actual_pos': actual_pos
        })
    
    def calibrate(self):
        """Perform calibration using collected data"""
        if len(self.calibration_data) < 4:  # Need at least 4 points
            return False
        
        # Calculate average offset and scaling
        gaze_points = np.array([d['gaze_point'] for d in self.calibration_data])
        actual_points = np.array([d['actual_pos'] for d in self.calibration_data])
        
        # Simple linear transformation (could be enhanced with more complex model)
        self.calibration_offset = np.mean(actual_points - gaze_points, axis=0)
        self.calibrated = True
        
        # Calculate calibration accuracy
        errors = []
        for d in self.calibration_data:
            adjusted = d['gaze_point'] + self.calibration_offset
            error = np.linalg.norm(adjusted - d['actual_pos'])
            errors.append(error)
        
        self.calibration_accuracy = 1 - (np.mean(errors) / 300)  # Rough accuracy metric
        return True
    
    def adjust_gaze_point(self, gaze_point):
        """Adjust gaze point using calibration data"""
        if not self.calibrated:
            return gaze_point
        return gaze_point + self.calibration_offset

class GazeKeyboardController:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = gazenet.GazeNet(device)
        self.face_detector = FaceDetector(device=device)
        self.keyboard = VirtualKeyboard()
        self.calibration_mode = False
        self.current_calibration_point = 0
        self.calibration_points = [
            (100, 100),  # Top-left
            (700, 100),  # Top-right
            (100, 400),  # Bottom-left
            (700, 400)   # Bottom-right
        ]
        
    def load_model(self, weights_path):
        """Load the gaze estimation model"""
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def run_calibration(self, frame):
        """Run the calibration procedure"""
        if self.current_calibration_point >= len(self.calibration_points):
            if self.keyboard.calibrate():
                self.calibration_mode = False
                print(f"Calibration complete! Estimated accuracy: {self.keyboard.calibration_accuracy:.2f}")
            return frame
        
        target = self.calibration_points[self.current_calibration_point]
        cv2.circle(frame, target, 20, (0, 0, 255), -1)
        cv2.putText(frame, "Look at the red dot", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Detect gaze and add calibration point when stable
        faces, landmarks = self.face_detector.detect(Image.fromarray(frame))
        if len(faces) > 0 and faces[0][-1] > 0.98:
            face, gaze_origin, M = utils.normalize_face(landmarks[0], frame)
            with torch.no_grad():
                gaze = self.model.get_gaze(face)
                gaze = gaze[0].data.cpu().numpy()
            
            # Project gaze to screen coordinates
            screen_distance = 1000
            scaling_factor = 100
            x_coord = int(gaze_origin[0] + gaze[0] * screen_distance / scaling_factor)
            y_coord = int(gaze_origin[1] + gaze[1] * screen_distance / scaling_factor)
            
            # Add to gaze history
            self.keyboard.gaze_history.append((x_coord, y_coord))
            
            # Check if gaze is stable
            if len(self.keyboard.gaze_history) == self.keyboard.gaze_history.maxlen:
                # Calculate variance of recent gaze points
                gaze_points = np.array(self.keyboard.gaze_history)
                variance = np.var(gaze_points, axis=0).mean()
                
                if variance < 50:  # Threshold for stable gaze
                    self.keyboard.add_calibration_point(
                        np.mean(gaze_points, axis=0),
                        target
                    )
                    self.current_calibration_point += 1
                    self.keyboard.gaze_history.clear()
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame"""
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        
        if self.calibration_mode:
            return self.run_calibration(display)
        
        # Detect faces and gaze
        faces, landmarks = self.face_detector.detect(Image.fromarray(display))
        if len(faces) > 0 and faces[0][-1] > 0.98:
            face, gaze_origin, M = utils.normalize_face(landmarks[0], display)
            with torch.no_grad():
                gaze = self.model.get_gaze(face)
                gaze = gaze[0].data.cpu().numpy()
            
            # Project gaze to screen coordinates
            screen_distance = 1000
            scaling_factor = 100
            x_coord = int(gaze_origin[0] + gaze[0] * screen_distance / scaling_factor)
            y_coord = int(gaze_origin[1] + gaze[1] * screen_distance / scaling_factor)
            
            # Adjust gaze point using calibration
            adjusted_gaze = self.keyboard.adjust_gaze_point((x_coord, y_coord))
            
            # Draw gaze point
            display = cv2.circle(display, (int(adjusted_gaze[0]), int(adjusted_gaze[1])), 5, (0, 0, 255), -1)
            
            # Check key press
            self.keyboard.check_key_press(adjusted_gaze)
        
        # Draw keyboard
        display = self.keyboard.draw_keyboard(display)
        
        # Draw instructions
        cv2.putText(display, "Press 'c' to calibrate, 'q' to quit", 
                    (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--weights', '-w', type=str, default='models/weights/gazenet.pth')
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    controller = GazeKeyboardController(device=device)
    controller.load_model(args.weights)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = frame[:, :, ::-1]  # BGR to RGB
            display = controller.process_frame(frame_rgb)
            
            cv2.imshow('Gaze Keyboard', cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('c'):
                controller.calibration_mode = True
                controller.current_calibration_point = 0
                controller.keyboard.calibration_data = []
                controller.keyboard.gaze_history.clear()
                print("Starting calibration...")
                
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()