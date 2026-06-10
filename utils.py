# utils.py
import time
import numpy as np

class FPSCounter:
    def __init__(self):
        self.start_time  = time.time()
        self.frame_count = 0
        self.fps         = 0

    def update(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.fps         = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time  = time.time()
        return self.fps

class AlertSystem:
    def __init__(self, eye_thresh=20, yawn_thresh=25):
        self.EYE_CONSEC_FRAMES  = eye_thresh
        self.YAWN_CONSEC_FRAMES = yawn_thresh
        self.eye_counter        = 0
        self.yawn_counter       = 0
        self.total_blinks       = 0
        self.total_yawns        = 0
        self.alert_active       = False

    def update(self, eyes_closed, mouth_open):
        if eyes_closed:
            self.eye_counter += 1
        else:
            if self.eye_counter >= 3:
                self.total_blinks += 1
            self.eye_counter = 0

        if mouth_open:
            self.yawn_counter += 1
        else:
            if self.yawn_counter >= self.YAWN_CONSEC_FRAMES:
                self.total_yawns += 1
            self.yawn_counter = 0

        if self.eye_counter >= self.EYE_CONSEC_FRAMES:
            self.alert_active = True
            return "danger"
        elif self.yawn_counter >= self.YAWN_CONSEC_FRAMES:
            self.alert_active = True
            return "warning"
        else:
            self.alert_active = False
            return "normal"

    def get_stats(self):
        return {
            "blinks"     : self.total_blinks,
            "yawns"      : self.total_yawns,
            "eye_frames" : self.eye_counter,
            "yawn_frames": self.yawn_counter
        }