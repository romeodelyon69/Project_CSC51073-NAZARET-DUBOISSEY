import cv2
from typing import Any
def display_ypr(yaw:float,pitch:float,roll:float,frame:Any,i:int,face_size:float):
    cv2.putText(
        frame,
        f"Yaw: {yaw:.2f}",
        (10, 30 + i * 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Pitch: {pitch:.2f}",
        (10, 60 + i * 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Roll: {roll:.2f}",
        (10, 90 + i * 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Face Size: {face_size:.4f}",
        (10, 120 + i * 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )