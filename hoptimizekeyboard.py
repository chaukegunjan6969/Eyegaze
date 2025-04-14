class OptimizedKeyboard:
    def __init__(self, screen_width=1280, screen_height=720):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.keys = self._create_optimized_layout()
        self.text = ""
        self.current_key = None
        self.dwell_start_time = 0
        self.dwell_threshold = 1.2  # seconds
        self.key_size = 80
        self.key_padding = 10
        self.keyboard_position = (screen_width - 700, 50)  # Right side of screen

    def _create_optimized_layout(self):
        """Create a keyboard layout optimized for gaze control"""
        # Main keys
        main_keys = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/']
        ]
        
        # Function keys
        function_keys = {
            'Space': (5, 3),  # Row, Col
            'Backspace': (0, 9),
            'Enter': (2, 9)
        }
        
        # Create key objects with positions
        keys = []
        start_x, start_y = self.keyboard_position
        
        for row_idx, row in enumerate(main_keys):
            for col_idx, key in enumerate(row):
                x = start_x + col_idx * (self.key_size + self.key_padding)
                y = start_y + row_idx * (self.key_size + self.key_padding)
                
                keys.append({
                    'label': key,
                    'rect': (x, y, x + self.key_size, y + self.key_size),
                    'center': (x + self.key_size//2, y + self.key_size//2)
                })
        
        # Add function keys
        for key, (row, col) in function_keys.items():
            x = start_x + col * (self.key_size + self.key_padding)
            y = start_y + row * (self.key_size + self.key_padding)
            
            # Make some keys larger
            width = self.key_size * 2 if key in ['Space', 'Backspace'] else self.key_size
            
            keys.append({
                'label': key,
                'rect': (x, y, x + width, y + self.key_size),
                'center': (x + width//2, y + self.key_size//2)
            })
        
        return keys

    def update(self, gaze_point):
        """Update keyboard state based on gaze"""
        self.current_key = None
        
        if gaze_point is None:
            return
            
        for key in self.keys:
            x1, y1, x2, y2 = key['rect']
            if x1 <= gaze_point[0] <= x2 and y1 <= gaze_point[1] <= y2:
                if self.current_key != key['label']:
                    self.current_key = key['label']
                    self.dwell_start_time = time.time()
                elif time.time() - self.dwell_start_time > self.dwell_threshold:
                    self._handle_key_press(key['label'])
                    self.current_key = None
                break

    def _handle_key_press(self, key):
        """Handle key press actions"""
        if key == 'Space':
            self.text += ' '
        elif key == 'Backspace':
            self.text = self.text[:-1]
        elif key == 'Enter':
            self.text += '\n'
        else:
            self.text += key

    def draw(self, frame):
        """Draw the keyboard on the frame"""
        # Draw text input area
        cv2.rectangle(frame, 
                     (self.keyboard_position[0], 10),
                     (self.keyboard_position[0] + 700, 40),
                     (240, 240, 240), -1)
        cv2.putText(frame, self.text, 
                   (self.keyboard_position[0] + 10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Draw keys
        for key in self.keys:
            x1, y1, x2, y2 = key['rect']
            
            # Highlight current key
            color = (200, 200, 255) if key['label'] == self.current_key else (255, 255, 255)
            
            # Draw key background
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            
            # Draw key label
            text_size = cv2.getTextSize(key['label'], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = x1 + ((x2 - x1) - text_size[0]) // 2
            text_y = y1 + ((y2 - y1) + text_size[1]) // 2
            cv2.putText(frame, key['label'], (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame