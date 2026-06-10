# main.py
# Real-Time Driver Attention Monitoring System
# Built by: Mohammed Haroon Khan
# Stack: Python, OpenCV, NumPy

import cv2
import numpy as np
import time
from utils    import FPSCounter, AlertSystem
from detector import FaceDetector

# ── THRESHOLDS ────────────────────────────────
EYE_OPEN_THRESHOLD = 0.08   # below = eye closed
EYE_FRAMES         = 20     # consecutive frames before alert
YAWN_FRAMES        = 25

# ── COLOURS ───────────────────────────────────
GREEN  = (0, 200,   0)
ORANGE = (0, 165, 255)
RED    = (0,   0, 255)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)

def draw_dashboard(frame, openness, alert_level, stats, fps):
    h, w   = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    alert_colors = {"normal": GREEN, "warning": ORANGE, "danger": RED}
    alert_color  = alert_colors.get(alert_level, GREEN)

    alert_text = {
        "normal" : "ATTENTIVE",
        "warning": "DROWSY WARNING!",
        "danger" : "WAKE UP!"
    }
    status = alert_text.get(alert_level, "ATTENTIVE")

    if alert_level != "normal":
        cv2.rectangle(frame, (0, h-60), (w, h), alert_color, -1)
        cv2.putText(frame, f"ALERT: {status}",
                    (w//2 - 160, h-15),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, WHITE, 2)

    cv2.putText(frame, f"Eye openness: {openness:.3f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                GREEN if openness > EYE_OPEN_THRESHOLD else RED, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)

    cv2.putText(frame,
                f"Blinks: {stats['blinks']}  Yawns: {stats['yawns']}",
                (w//2-110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    cv2.circle(frame, (w-25, 60), 12, alert_color, -1)
    return frame

def main():
    print("="*50)
    print("  Driver Attention Monitoring System")
    print("  Built by: Mohammed Haroon Khan")
    print("="*50)

    detector     = FaceDetector()
    alert_system = AlertSystem(EYE_FRAMES, YAWN_FRAMES)
    fps_counter  = FPSCounter()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No camera found")
        return

    print("[INFO] Camera ready — press Q to quit, S for screenshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        landmarks = detector.get_landmarks(frame)

        if landmarks:
            eyes      = landmarks["eyes"]
            face_gray = landmarks["face_gray"]

            # Calculate openness for both eyes
            openness_vals = [
                detector.calculate_eye_openness(face_gray, eye)
                for eye in eyes
            ]
            openness    = sum(openness_vals) / len(openness_vals)
            eyes_closed = openness < EYE_OPEN_THRESHOLD
            alert_level = alert_system.update(eyes_closed, False)
            frame       = detector.draw_landmarks(frame, landmarks)
        else:
            openness    = 0.5
            alert_level = "normal"
            alert_system.update(False, False)
            cv2.putText(frame, "No face detected",
                        (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, ORANGE, 2)

        stats = alert_system.get_stats()
        fps   = fps_counter.update()
        frame = draw_dashboard(frame, openness,
                               alert_level, stats, fps)

        cv2.imshow("Driver Attention Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()

    stats = alert_system.get_stats()
    print("\n" + "="*50)
    print("  Session Summary")
    print("="*50)
    print(f"  Total blinks : {stats['blinks']}")
    print(f"  Total yawns  : {stats['yawns']}")
    print("="*50)

if __name__ == "__main__":
    main()