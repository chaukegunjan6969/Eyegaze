import cv2
import numpy as np
import dlib
from sklearn.linear_model import LinearRegression
import time
from collections import deque

class VirtualKeyboard:
    def __init__(self, rows=3, cols=10):
        self.rows = rows
        self.cols = cols
        self.keys = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/']
        ]
        self.key_regions = []
        self.key_size = (60, 60)
        self.key_padding = 5
        self.keyboard_pos = (50, 150)
        self.text = ""
        self.text_pos = (50, 100)
        self.current_key = None
        self.key_selection_start = None
        self.dwell_time_threshold = 1.5  # seconds
    
    def draw_keyboard(self, frame):
        key_w, key_h = self.key_size
        padding = self.key_padding
        start_x, start_y = self.keyboard_pos
        
        self.key_regions = []
        
        # Text display area
        cv2.rectangle(frame, (self.text_pos[0], self.text_pos[1] - 30), 
                     (self.text_pos[0] + 600, self.text_pos[1] + 10), 
                     (200, 200, 200), -1)
        cv2.putText(frame, self.text, (self.text_pos[0] + 5, self.text_pos[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Draw keys
        for row in range(self.rows):
            for col in range(self.cols):
                x1 = start_x + col * (key_w + padding)
                y1 = start_y + row * (key_h + padding)
                x2 = x1 + key_w
                y2 = y1 + key_h
                
                self.key_regions.append({
                    'key': self.keys[row][col],
                    'rect': (x1, y1, x2, y2),
                    'center': ((x1+x2)//2, (y1+y2)//2)
                })
                
                color = (255, 255, 255)
                if self.current_key == self.keys[row][col]:
                    color = (0, 255, 0)  # Highlight current key
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                
                text_size = cv2.getTextSize(self.keys[row][col], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = x1 + (key_w - text_size[0]) // 2
                text_y = y1 + (key_h + text_size[1]) // 2
                cv2.putText(frame, self.keys[row][col], (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return frame
    
    def update_gaze_point(self, gaze_point):
        if not self.key_regions:
            return
        
        new_key = None
        for key_data in self.key_regions:
            x1, y1, x2, y2 = key_data['rect']
            if x1 <= gaze_point[0] <= x2 and y1 <= gaze_point[1] <= y2:
                new_key = key_data['key']
                break
        
        if new_key != self.current_key:
            self.current_key = new_key
            self.key_selection_start = time.time() if new_key else None
        elif new_key and (time.time() - self.key_selection_start) > self.dwell_time_threshold:
            self.text += new_key
            self.current_key = None
            self.key_selection_start = None

class EnhancedGazeTracker:
    def __init__(self, screen_width=1280, screen_height=720):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.gaze_mapper = LinearRegression()
        self.calibration_data = []
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.calibration_points = [
            (100, 100), (screen_width//2, 100), (screen_width-100, 100),
            (100, screen_height//2), (screen_width//2, screen_height//2), (screen_width-100, screen_height//2),
            (100, screen_height-100), (screen_width//2, screen_height-100), (screen_width-100, screen_height-100)
        ]
        self.current_calibration_point = 0
        self.calibration_complete = False
        self.gaze_history = deque(maxlen=30)
        self.last_gaze_point = None
        self.error_message = ""
        self.error_timer = 0
    
    def detect_eyes(self, landmarks):
        left_eye_points = landmarks[36:42]
        right_eye_points = landmarks[42:48]
        
        left_center = np.mean(left_eye_points, axis=0)
        right_center = np.mean(right_eye_points, axis=0)
        
        return {
            'left_eye': {'center': left_center, 'points': left_eye_points},
            'right_eye': {'center': right_center, 'points': right_eye_points}
        }
    
    def estimate_head_pose(self, landmarks):
        model_points = np.array([
            (0.0, 0.0, 0.0),         # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ])
        
        image_points = np.array([
            landmarks[30], landmarks[8], landmarks[36],
            landmarks[45], landmarks[48], landmarks[54]
        ], dtype="double")
        
        focal_length = self.screen_width
        center = (self.screen_width/2, self.screen_height/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        dist_coeffs = np.zeros((4,1))
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        return rotation_vector.flatten(), translation_vector.flatten()
    
    def calibrate(self, frame):
        if self.current_calibration_point >= len(self.calibration_points):
            if len(self.calibration_data) >= 9:
                self.train_calibration_model()
                self.calibration_complete = True
                self.error_message = ""
                return None, True
            return None, False
        
        target = self.calibration_points[self.current_calibration_point]
        cv2.circle(frame, target, 20, (0, 0, 255), -1)
        cv2.putText(frame, "Look at the red dot", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            self.error_message = "ERROR: No face detected!"
            self.error_timer = time.time()
            return target, False
        
        landmarks = self.predictor(gray, faces[0])
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        eye_features = self.detect_eyes(landmarks)
        head_rotation, head_translation = self.estimate_head_pose(landmarks)
        
        left_pos = eye_features['left_eye']['center']
        right_pos = eye_features['right_eye']['center']
        
        # Normalize eye positions
        left_norm = (left_pos[0]/self.screen_width, left_pos[1]/self.screen_height)
        right_norm = (right_pos[0]/self.screen_width, right_pos[1]/self.screen_height)
        
        # Add to gaze history
        self.gaze_history.append({
            'left_eye': left_norm,
            'right_eye': right_norm,
            'head_rot': head_rotation,
            'head_trans': head_translation
        })
        
        # Check stability
        if len(self.gaze_history) == self.gaze_history.maxlen:
            left_x = [g['left_eye'][0] for g in self.gaze_history]
            left_y = [g['left_eye'][1] for g in self.gaze_history]
            var = np.var(left_x) + np.var(left_y)
            
            if var < 0.001:  # Stable gaze threshold
                avg_left = np.mean([g['left_eye'] for g in self.gaze_history], axis=0)
                avg_right = np.mean([g['right_eye'] for g in self.gaze_history], axis=0)
                avg_rot = np.mean([g['head_rot'] for g in self.gaze_history], axis=0)
                avg_trans = np.mean([g['head_trans'] for g in self.gaze_history], axis=0)
                
                self.calibration_data.append({
                    'features': [*avg_left, *avg_right, *avg_rot, *avg_trans],
                    'target': target
                })
                self.current_calibration_point += 1
                self.gaze_history.clear()
                self.error_message = ""
        
        return target, False
    
    def train_calibration_model(self):
        X = [d['features'] for d in self.calibration_data]
        y = [d['target'] for d in self.calibration_data]
        self.gaze_mapper.fit(X, y)
    
    def map_gaze(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            self.error_message = "ERROR: No face detected!"
            self.error_timer = time.time()
            return None
        
        landmarks = self.predictor(gray, faces[0])
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        eye_features = self.detect_eyes(landmarks)
        head_rotation, head_translation = self.estimate_head_pose(landmarks)
        
        left_pos = eye_features['left_eye']['center']
        right_pos = eye_features['right_eye']['center']
        
        left_norm = (left_pos[0]/self.screen_width, left_pos[1]/self.screen_height)
        right_norm = (right_pos[0]/self.screen_width, right_pos[1]/self.screen_height)
        
        features = [*left_norm, *right_norm, *head_rotation, *head_translation]
        
        if self.calibration_complete:
            try:
                gaze_point = self.gaze_mapper.predict([features])[0]
                self.last_gaze_point = (int(gaze_point[0]), int(gaze_point[1]))
                self.error_message = ""
                return self.last_gaze_point
            except:
                self.error_message = "ERROR: Gaze mapping failed!"
                self.error_timer = time.time()
                return None
        else:
            # Fallback without calibration
            avg_x = (left_pos[0] + right_pos[0]) / 2
            avg_y = (left_pos[1] + right_pos[1]) / 2
            self.last_gaze_point = (int(avg_x), int(avg_y))
            return self.last_gaze_point
    
    def draw_gaze_visualization(self, frame):
        # Draw last known gaze point
        if self.last_gaze_point:
            cv2.circle(frame, self.last_gaze_point, 10, (0, 0, 255), -1)
            cv2.putText(frame, "Gaze Point", 
                       (self.last_gaze_point[0] + 15, self.last_gaze_point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Show error message if any (for 3 seconds)
        if self.error_message and (time.time() - self.error_timer) < 3:
            cv2.putText(frame, self.error_message, (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def get_camera():
    # Try different backends
    backends = [
        cv2.CAP_DSHOW,  # Works best on Windows
        cv2.CAP_MSMF,
        cv2.CAP_V4L2,
        cv2.CAP_ANY
    ]
    
    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            print(f"Using backend: {backend}")
            # Set reasonable resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
    
    raise RuntimeError("Could not open camera with any backend")

def main():
    try:
        cap = get_camera()
    except RuntimeError as e:
        print(e)
        return
    
    gaze_tracker = EnhancedGazeTracker()
    keyboard = VirtualKeyboard()
    
    calibration_mode = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Couldn't read frame from camera")
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            
            if calibration_mode:
                target, complete = gaze_tracker.calibrate(display)
                if complete:
                    calibration_mode = False
                    print("Calibration complete! Starting keyboard...")
            else:
                gaze_point = gaze_tracker.map_gaze(frame)
                if gaze_point is not None:
                    keyboard.update_gaze_point(gaze_point)
                
                display = keyboard.draw_keyboard(display)
            
            # Draw gaze visualization and errors
            gaze_tracker.draw_gaze_visualization(display)
            
            # Show instructions
            if calibration_mode:
                cv2.putText(display, f"Calibration Point {gaze_tracker.current_calibration_point + 1}/9", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Using keyboard - Press 'r' to recalibrate", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Gaze Keyboard", display)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                calibration_mode = True
                gaze_tracker.current_calibration_point = 0
                gaze_tracker.calibration_data = []
                gaze_tracker.calibration_complete = False
                gaze_tracker.gaze_history.clear()
                gaze_tracker.error_message = ""
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()