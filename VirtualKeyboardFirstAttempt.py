import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Virtual Keyboard Class (Improved)
class EyeTrackingKeyboard:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Keyboard layout
        self.keys = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', '←'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', ' ']
        ]
        
        # Keyboard parameters
        self.key_width = 60
        self.key_height = 80
        self.key_spacing = 5
        self.start_x = screen_width - (len(self.keys[0]) * (self.key_width + self.key_spacing)) // 2
        self.start_y = screen_height - 300
        self.key_positions = self._calculate_key_positions()
        self.text = ""
        self.selection_threshold = 1.0  # seconds
        self.current_gaze = None
        self.gaze_history = deque(maxlen=10)  # Smoothing gaze points
        
        # Eye tracking parameters
        self.eye_landmarks = {
            'left': [33, 133, 160, 144, 158, 153, 157, 154],  # Left eye landmarks
            'right': [362, 263, 385, 380, 373, 374, 390, 249]  # Right eye landmarks
        }
        
        # Calibration
        self.calibration_points = []
        self.calibration_data = []
        self.is_calibrating = False
        self.calibration_index = 0
        self.calibration_complete = False
        self.calibration_targets = [
            (screen_width // 2, screen_height // 2),  # Center
            (50, 50),  # Top-left
            (screen_width - 50, 50),  # Top-right
            (50, screen_height - 50),  # Bottom-left
            (screen_width - 50, screen_height - 50)  # Bottom-right
        ]
        
    def _calculate_key_positions(self):
        positions = {}
        for row_idx, row in enumerate(self.keys):
            for col_idx, key in enumerate(row):
                x = self.start_x + col_idx * (self.key_width + self.key_spacing)
                y = self.start_y + row_idx * (self.key_height + self.key_spacing)
                positions[key] = (x, y, x + self.key_width, y + self.key_height)
        return positions
    
    def draw(self, frame):
        # Draw semi-transparent keyboard background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, self.start_y - 20), 
                      (self.screen_width, self.screen_height), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw keys
        for key, (x1, y1, x2, y2) in self.key_positions.items():
            color = (100, 100, 100)  # Default gray
            
            # Highlight if being looked at
            if self.current_gaze is not None and self._is_gaze_on_key(key, self.current_gaze):
                color = (0, 150, 255)  # Orange
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
            
            # Draw key label
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = x1 + (self.key_width - text_size[0]) // 2
            text_y = y1 + (self.key_height + text_size[1]) // 2
            cv2.putText(frame, key, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw text input area
        cv2.rectangle(frame, (self.start_x, self.start_y - 70), 
                     (self.start_x + len(self.keys[0]) * (self.key_width + self.key_spacing), 
                      self.start_y - 20), (70, 70, 70), -1)
        cv2.putText(frame, self.text, (self.start_x + 10, self.start_y - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def _is_gaze_on_key(self, key, gaze_point):
        x1, y1, x2, y2 = self.key_positions[key]
        return x1 <= gaze_point[0] <= x2 and y1 <= gaze_point[1] <= y2
    
    def update_gaze(self, gaze_point):
        if gaze_point is not None:  # Changed from 'if gaze_point:'
            self.gaze_history.append(gaze_point)
            if self.gaze_history:  # Only calculate mean if history is not empty
                self.current_gaze = np.mean(self.gaze_history, axis=0)
    
    def check_selection(self):
        if self.current_gaze is None:
            return False
            
        for key in self.key_positions:
            if self._is_gaze_on_key(key, self.current_gaze):
                # Check if gaze has been stable long enough
                stable_count = sum(1 for gaze in self.gaze_history 
                                 if self._is_gaze_on_key(key, gaze))
                
                if stable_count / len(self.gaze_history) > 0.8:  # 80% of recent gaze points on key
                    self._process_key_selection(key)
                    return True
        return False
    
    def _process_key_selection(self, key):
        if key == '←':
            self.text = self.text[:-1]  # Backspace
        elif key == ' ':
            self.text += ' '  # Space
        else:
            self.text += key.lower()  # Normal key

# Main Application
def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        return
    
    screen_height, screen_width = frame.shape[:2]
    keyboard = EyeTrackingKeyboard(screen_width, screen_height)
    
    # Calibration variables
    calibration_start_time = 0
    calibration_collected = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        gaze_point = None
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get eye landmarks
            left_eye_landmarks = np.array([(face_landmarks.landmark[i].x * screen_width, 
                                          face_landmarks.landmark[i].y * screen_height) 
                                         for i in keyboard.eye_landmarks['left']])
            
            right_eye_landmarks = np.array([(face_landmarks.landmark[i].x * screen_width, 
                                           face_landmarks.landmark[i].y * screen_height) 
                                          for i in keyboard.eye_landmarks['right']])
            
            # Calculate eye centers
            left_eye_center = np.mean(left_eye_landmarks, axis=0)
            right_eye_center = np.mean(right_eye_landmarks, axis=0)
            
            # Calculate gaze point (average of both eyes)
            gaze_point = ((left_eye_center + right_eye_center) / 2)
        
        # Update keyboard with current gaze point
        keyboard.update_gaze(gaze_point)
        
        # Handle calibration
        if keyboard.is_calibrating:
            target_point = keyboard.calibration_targets[keyboard.calibration_index]
            cv2.circle(frame, target_point, 20, (0, 255, 255), -1)
            cv2.putText(frame, "Look at the yellow dot", 
                        (screen_width // 2 - 100, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            if gaze_point is not None:
                distance = np.linalg.norm(np.array(gaze_point) - np.array(target_point))
                
                if distance < 50:  # pixels threshold
                    if not calibration_collected:
                        calibration_start_time = time.time()
                        calibration_collected = True
                    
                    if time.time() - calibration_start_time > 1.0:  # Collect for 1 second
                        keyboard.calibration_points.append((gaze_point, target_point))
                        keyboard.calibration_index += 1
                        calibration_collected = False
                        
                        if keyboard.calibration_index >= len(keyboard.calibration_targets):
                            keyboard._complete_calibration()
                else:
                    calibration_collected = False
        else:
            # Draw keyboard and handle input
            frame = keyboard.draw(frame)
            keyboard.check_selection()
        
        # Display status
        status_text = [
            f"Calibration: {'Complete' if keyboard.calibration_complete else 'Incomplete'}",
            f"Text: {keyboard.text}",
            "Press 'c' to calibrate, 'q' to quit"
        ]
        
        for i, text in enumerate(status_text):
            cv2.putText(frame, text, (10, 30 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Eye Tracking Keyboard', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c') and not keyboard.is_calibrating:
            keyboard.is_calibrating = True
            keyboard.calibration_index = 0
            keyboard.calibration_points = []
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()