import csv
import time
from typing import Any
from pathlib import Path
from landmarks import (
    NOSE_LANDMARK,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    LEFT_EYE_INNER,
    RIGHT_EYE_INNER,
    LEFT_EYE_TOP,
    RIGHT_EYE_TOP,
    LEFT_EYE_BOTTOM,
    RIGHT_EYE_BOTTOM,
)

DATA_FOLDER = Path("data")


class CSVWriter:
    def __init__(self):
        if not DATA_FOLDER.exists():
            DATA_FOLDER.mkdir(parents=True, exist_ok=True)
        self.filename = DATA_FOLDER / ("face_data" + str(int(time.time())) + ".csv")
        self.csv_writer = csv.writer(open(self.filename, mode="w", newline=""))
        self.csv_writer.writerow(
            [
                "nose X",
                "nose Y",
                "left eye X",
                "left eye Y",
                "right eye X",
                "right eye Y",
                "left eye bottom X",
                "left eye bottom Y",
                "right eye bottom X",
                "right eye bottom Y",
                "left eye top X",
                "left eye top Y",
                "right eye top X",
                "right eye top Y",
                "left eye inner X",
                "left eye inner Y",
                "right eye inner X",
                "right eye inner Y",
                "left eye outer X",
                "left eye outer Y",
                "right eye outer X",
                "right eye outer Y",
                "left_pupil X",
                "left_pupil Y",
                "right_pupil X",
                "right_pupil Y",
                "Yaw",
                "Pitch",
                "Roll",
                "Face Size",
                "target X normalized",
                "target Y normalized",
            ]
        )

    def log_step(
        self,
        landmarks: Any,
        x_left_eye: float,
        y_left_eye: float,
        x_right_eye: float,
        y_right_eye: float,
        x_left_pupil: float,
        y_left_pupil: float,
        x_right_pupil: float,
        y_right_pupil: float,
        yaw: float,
        pitch: float,
        roll: float,
        face_size: float,
        cross_position_x: float,
        cross_position_y: float,
        w: int,
        h: int,
        screen_width: int,
        screen_height: int,
    ):
        noseX = landmarks.landmark[NOSE_LANDMARK].x
        noseY = landmarks.landmark[NOSE_LANDMARK].y
        leftEyeX = x_left_eye / w
        leftEyeY = y_left_eye / h
        rightEyeX = x_right_eye / w
        rightEyeY = y_right_eye / h
        leftPupilX = x_left_pupil / w
        leftPupilY = y_left_pupil / h
        rightPupilX = x_right_pupil / w
        rightPupilY = y_right_pupil / h
        leftEyeOuterX = landmarks.landmark[LEFT_EYE_OUTER].x
        leftEyeOuterY = landmarks.landmark[LEFT_EYE_OUTER].y
        rightEyeOuterX = landmarks.landmark[RIGHT_EYE_OUTER].x
        rightEyeOuterY = landmarks.landmark[RIGHT_EYE_OUTER].y
        leftEyeInnerX = landmarks.landmark[LEFT_EYE_INNER].x
        leftEyeInnerY = landmarks.landmark[LEFT_EYE_INNER].y
        rightEyeInnerX = landmarks.landmark[RIGHT_EYE_INNER].x
        rightEyeInnerY = landmarks.landmark[RIGHT_EYE_INNER].y
        leftEyeTopX = landmarks.landmark[LEFT_EYE_TOP].x
        leftEyeTopY = landmarks.landmark[LEFT_EYE_TOP].y
        rightEyeTopX = landmarks.landmark[RIGHT_EYE_TOP].x
        rightEyeTopY = landmarks.landmark[RIGHT_EYE_TOP].y
        leftEyeBottomX = landmarks.landmark[LEFT_EYE_BOTTOM].x
        leftEyeBottomY = landmarks.landmark[LEFT_EYE_BOTTOM].y
        rightEyeBottomX = landmarks.landmark[RIGHT_EYE_BOTTOM].x
        rightEyeBottomY = landmarks.landmark[RIGHT_EYE_BOTTOM].y

        # compute the pupil position in the same frame of reference as the eye position
        self.csv_writer.writerow(
            [
                noseX,
                noseY,
                leftEyeX,
                leftEyeY,
                rightEyeX,
                rightEyeY,
                leftEyeBottomX,
                leftEyeBottomY,
                rightEyeBottomX,
                rightEyeBottomY,
                leftEyeTopX,
                leftEyeTopY,
                rightEyeTopX,
                rightEyeTopY,
                leftEyeInnerX,
                leftEyeInnerY,
                rightEyeInnerX,
                rightEyeInnerY,
                leftEyeOuterX,
                leftEyeOuterY,
                rightEyeOuterX,
                rightEyeOuterY,
                leftPupilX,
                leftPupilY,
                rightPupilX,
                rightPupilY,
                yaw,
                pitch,
                roll,
                face_size,
                (cross_position_x) / screen_width,
                (cross_position_y) / screen_height,
            ]
        )
