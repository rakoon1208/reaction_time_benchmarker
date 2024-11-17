import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        # Initialize MediaPipe Hand solutions
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,  # Allow detection of two hands
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        """
        Detect hands and optionally draw landmarks on the frame.
        :param frame: The input frame from the video feed.
        :param draw: Boolean indicating whether to draw landmarks.
        :return: Processed frame with or without landmarks drawn.
        """
        # Convert the frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)

        # Draw landmarks if requested
        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return frame

    def get_hand_bounding_boxes(self, frame):
        """
        Get bounding boxes for all detected hands.
        :param frame: The input frame from the video feed.
        :return: List of bounding boxes (each as (x, y, width, height)) for each detected hand.
        """
        bounding_boxes = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
                y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                bounding_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
        return bounding_boxes

    def close(self):
        """Release resources used by MediaPipe Hand solution."""
        self.hands.close()
