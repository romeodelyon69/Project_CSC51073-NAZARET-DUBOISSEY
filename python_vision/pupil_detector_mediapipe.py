import csv
from time import time
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import eyeToolKit as etk
import math
import time
import Nlib
import faceMath 
import test_world_landmarks as faceCompute

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)


LEFT_IRIS_LANDMARKS = [474, 475, 477, 476]  # Left iris landmarks
RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]  # Right iris landmarks

NOSE_LANDMARKS = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

CHIN_LANDMARK = 152
NOSE_LANDMARK = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
LEFT_EYE_TOP = 159
RIGHT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 23
RIGHT_EYE_BOTTOM = 253
LEFT_PUPIL = 473
RIGHT_PUPIL = 468
LEFT_MOUTH = 78
RIGHT_MOUTH = 308

LEFT_EYE_PUPIL = 468
RIGHT_EYE_PUPIL = 473

SCREEN_WIDTH = 1540
SCREEN_HEIGHT = 880
# open a csv file to save the face data

# record video from webcam in .MOV format
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (640, 480))

left_bufferX = Nlib.Buffer(9)
left_bufferY = Nlib.Buffer(9)
right_bufferX = Nlib.Buffer(9)
right_bufferY = Nlib.Buffer(9)

face = faceMath.face(None, {"width": 640, "height": 480, "focal_length": 1})

#load camera calibration parameters from file 
file_path = "calibration_data.npz"
with np.load(file_path) as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

face2 = faceCompute.face(None, {"width": 640, "height": 480}, cam_matrix=camera_matrix)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    h, w, _ = frame.shape
    face_orientation = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_orientation.append(
                etk.get_face_orientation(
                    face_landmarks.landmark,
                    CHIN_LANDMARK,
                    NOSE_LANDMARK,
                    LEFT_EYE_OUTER,
                    RIGHT_EYE_OUTER,
                    LEFT_MOUTH,
                    RIGHT_MOUTH,
                )
            )
            face_size = etk.get_face_size(
                face_landmarks.landmark,
                CHIN_LANDMARK,
                NOSE_LANDMARK,
                LEFT_EYE_OUTER,
                RIGHT_EYE_OUTER,
            )

        # face.landmarks = face_landmarks.landmark
        # face.compute_face_orientation(frame, NOSE_LANDMARKS, color=(0, 255, 0), size=80)
        # face.compute_eye_vectors()
        # face.compute_gaze_intersection()

        # face.draw_pupil_positions(frame)
        # face.draw_face_orientation(frame, size=80, color=(0, 255, 0))
        # face.draw_face_landmarks(frame, NOSE_LANDMARKS, color=(0, 255, 0))
        # face.draw_debug_interface()
        face2.landmarks = face_landmarks.landmark
        face2.update()
        face2.draw_world_landmarks()

        
        if cv2.waitKey(1) & 0xFF == ord("r"):
            face2.save_eyeball_reference()
            print("Eyeball reference saved.")


        # Agrandit l'image 4x
        frame_big = cv2.resize(
            frame, (0, 0), fx=1.6, fy=1.6, interpolation=cv2.INTER_NEAREST
        )

        # show the result
        cv2.imshow("Eye Tracking", frame_big)
        # cv2.imshow("Extracted Eyes", eye_cropped_big if 'eye_cropped_big' in locals() else np.zeros((100, 100, 3), dtype=np.uint8))
        # cv2.imshow("Extracted Eyes Red", eye_cropped_big_without if 'eye_cropped_big_without' in locals() else np.zeros((100, 100, 3), dtype=np.uint8))

    # record video from webcam
    out.write(frame)
    # print("frame size :", frame.shape)

    if cv2.waitKey(1) & 0xFF == 27:  # Échap pour quitter
        break


cap.release()
out.release()
cv2.destroyAllWindows()
