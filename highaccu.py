import cv2
import numpy as np
import dlib
from sklearn.ensemble import RandomForestRegressor
import time
from collections import deque

class HighAccuracyGazeTracker:
    def __init__(self, screen_width=1280, screen_height=720):
        # Initialize with higher quality models
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Use more sophisticated model for gaze mapping
        self.gaze_model = RandomForestRegressor(n_estimators=100, max_depth=10)
        
        # Screen configuration
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Calibration setup
        self.calibration_points = self._generate_calibration_grid()
        self.current_calibration_index = 0
        self.calibration_data = []
        self.is_calibrated = False
        
        # Tracking variables
        self.last_gaze_point = None
        self.gaze_history = deque(maxlen=15)  # For smoothing
        self.error_message = ""
        self.error_timer = 0

    def _generate_calibration_grid(self, rows=5, cols=5):
        """Generate more calibration points for better accuracy"""
        points = []
        x_step = self.screen_width // (cols + 1)
        y_step = self.screen_height // (rows + 1)
        
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                points.append((col * x_step, row * y_step))
        
        return points

    def _extract_eye_features(self, landmarks):
        """Enhanced feature extraction with more detailed eye metrics"""
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        # Calculate multiple features per eye
        def eye_metrics(eye_points):
            center = np.mean(eye_points, axis=0)
            width = np.linalg.norm(eye_points[3] - eye_points[0])
            height = np.linalg.norm(eye_points[1] - eye_points[5])
            return {
                'center': center,
                'width': width,
                'height': height,
                'aspect_ratio': width / height,
                'contour': eye_points
            }
        
        return {
            'left_eye': eye_metrics(left_eye),
            'right_eye': eye_metrics(right_eye)
        }

    def calibrate(self, frame):
        """Enhanced calibration with more points and quality checks"""
        target = self.calibration_points[self.current_calibration_index]
        
        # Draw calibration UI
        cv2.circle(frame, target, 15, (0, 0, 255), -1)
        cv2.putText(frame, f"Calibration Point {self.current_calibration_index + 1}/{len(self.calibration_points)}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Detect face and features
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        if len(faces) == 0:
            self._show_error("Face not detected")
            return False
        
        landmarks = self.landmark_predictor(gray, faces[0])
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        eye_features = self._extract_eye_features(landmarks)
        
        # Store calibration data with timestamp
        self.calibration_data.append({
            'timestamp': time.time(),
            'eye_features': eye_features,
            'head_pose': self._estimate_head_pose(landmarks),
            'target': target
        })
        
        # Progress to next point after short delay
        if time.time() - self.calibration_data[-1]['timestamp'] > 2.0:
            self.current_calibration_index += 1
            
            if self.current_calibration_index >= len(self.calibration_points):
                self._train_gaze_model()
                return True
        
        return False

    def _train_gaze_model(self):
        """Train the model with calibration data"""
        X = []
        y = []
        
        for data in self.calibration_data:
            features = self._create_feature_vector(
                data['eye_features'], 
                data['head_pose']
            )
            X.append(features)
            y.append(data['target'])
        
        self.gaze_model.fit(X, y)
        self.is_calibrated = True
        print("Calibration complete! Model trained with", len(X), "samples.")

    def predict_gaze(self, frame):
        """Predict gaze position with error handling"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)
            
            if len(faces) == 0:
                self._show_error("Face not detected")
                return None
                
            landmarks = self.landmark_predictor(gray, faces[0])
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            eye_features = self._extract_eye_features(landmarks)
            head_pose = self._estimate_head_pose(landmarks)
            
            features = self._create_feature_vector(eye_features, head_pose)
            
            if self.is_calibrated:
                prediction = self.gaze_model.predict([features])[0]
                self.last_gaze_point = (int(prediction[0]), int(prediction[1]))
                
                # Apply temporal smoothing
                self.gaze_history.append(self.last_gaze_point)
                smoothed_point = np.mean(self.gaze_history, axis=0)
                return (int(smoothed_point[0]), int(smoothed_point[1]))
            else:
                # Fallback to simple center between eyes
                left = eye_features['left_eye']['center']
                right = eye_features['right_eye']['center']
                return (int((left[0] + right[0]) / 2), int((left[1] + right[1]) / 2))
                
        except Exception as e:
            self._show_error(f"Prediction error: {str(e)}")
            return None

    def _create_feature_vector(self, eye_features, head_pose):
        """Create comprehensive feature vector"""
        left = eye_features['left_eye']
        right = eye_features['right_eye']
        
        return [
            # Left eye features
            left['center'][0], left['center'][1],
            left['width'], left['height'],
            left['aspect_ratio'],
            
            # Right eye features
            right['center'][0], right['center'][1],
            right['width'], right['height'],
            right['aspect_ratio'],
            
            # Relative positions
            left['center'][0] - right['center'][0],
            left['center'][1] - right['center'][1],
            
            # Head pose
            *head_pose[0],  # rotation
            *head_pose[1]   # translation
        ]

    def _estimate_head_pose(self, landmarks):
        """More robust head pose estimation"""
        # 3D model points (generic head model)
        model_points = np.array([
            (0.0, 0.0, 0.0),         # Nose tip
            (0.0, -330.0, -65.0),    # Chin
            (-225.0, 170.0, -135.0), # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),# Left mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])
        
        # Image points from landmarks
        image_points = np.array([
            landmarks[30], landmarks[8], landmarks[36],
            landmarks[45], landmarks[48], landmarks[54]
        ], dtype="double")
        
        # Camera parameters
        focal_length = self.screen_width
        center = (self.screen_width/2, self.screen_height/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        dist_coeffs = np.zeros((4,1))
        
        # Solve for pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE)
        
        return rotation_vector.flatten(), translation_vector.flatten()

    def _show_error(self, message):
        """Display error message"""
        self.error_message = message
        self.error_timer = time.time()