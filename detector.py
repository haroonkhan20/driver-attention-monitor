# detector.py
# Face detection using OpenCV only — no MediaPipe needed

import cv2
import numpy as np

class FaceDetector:
    """
    Detects face and eyes using OpenCV Haar Cascades.
    No MediaPipe, no TensorFlow — works on any Python setup.
    """

    def __init__(self):
        # Built into OpenCV — no extra install needed
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml")
        self.available    = True
        print("[Detector] OpenCV Haar Cascade loaded")

    def get_landmarks(self, frame):
        """
        Detect face and eyes.
        Returns eye regions or None if no face found.
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) == 0:
            return None

        # Take the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_gray   = gray[y:y+h, x:x+w]
        face_color  = frame[y:y+h, x:x+w]

        # Detect eyes inside face region
        eyes = self.eye_cascade.detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=5)

        if len(eyes) < 2:
            return None

        # Sort eyes left to right
        eyes = sorted(eyes, key=lambda e: e[0])

        return {
            "face"      : (x, y, w, h),
            "eyes"      : eyes[:2],
            "face_gray" : face_gray,
            "frame_y"   : y,
            "frame_x"   : x
        }

    def calculate_eye_openness(self, face_gray, eye):
        """
        Estimate eye openness using pixel intensity.
        Closed eye = more uniform dark pixels.
        Returns float 0-1 (higher = more open)
        """
        ex, ey, ew, eh = eye
        eye_region = face_gray[ey:ey+eh, ex:ex+ew]

        if eye_region.size == 0:
            return 0.5

        # Resize for consistency
        eye_region = cv2.resize(eye_region, (24, 12))

        # Detect edges — open eye has more edges (eyelashes, iris)
        edges     = cv2.Canny(eye_region, 50, 150)
        openness  = np.sum(edges > 0) / edges.size
        return round(openness, 4)

    def draw_landmarks(self, frame, landmark_data):
        """Draw face and eye boxes."""
        if not landmark_data:
            return frame

        x, y, w, h = landmark_data["face"]
        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Draw eye boxes
        for (ex, ey, ew, eh) in landmark_data["eyes"]:
            cv2.rectangle(frame,
                          (x+ex, y+ey),
                          (x+ex+ew, y+ey+eh),
                          (255, 0, 0), 2)
        return frame