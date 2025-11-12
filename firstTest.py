import csv
from time import time
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import eyeToolKit as etk
import math
import MlTools as mlt
import torch
import time
import Nlib

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True
)

cap = cv2.VideoCapture(0)

LEFT_EYE_LANDMARKS = [
    463,
    398,
    384,
    385,
    386,
    387,
    388,
    466,
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
]  # Left eye landmarks
RIGHT_EYE_LANDMARKS = [
    33,
    246,
    161,
    160,
    159,
    158,
    157,
    173,
    133,
    155,
    154,
    153,
    145,
    144,
    163,
    7,
]  # Right eye landmarks

LEFT_IRIS_LANDMARKS = [474, 475, 477, 476]  # Left iris landmarks
RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]  # Right iris landmarks

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

LEFT_EYE_CENTER = 468
RIGHT_EYE_CENTER = 473

SCREEN_WIDTH = 1540
SCREEN_HEIGHT = 880
# open a csv file to save the face data

cross_position_x = 0
cross_position_y = 0
speed = 20  # pixels per frame


filename = "face_data" + str(int(time.time())) + ".csv"
csv_file = open(filename, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
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

# record video from webcam in .MOV format
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (640, 480))

left_bufferX = Nlib.Buffer(9)
left_bufferY = Nlib.Buffer(9)
right_bufferX = Nlib.Buffer(9)
right_bufferY = Nlib.Buffer(9)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    h, w, _ = frame.shape
    print("h:", h, "w:", w)

    face_orientation = []
    tick = cv2.getTickCount()
    tick = tick / cv2.getTickFrequency()  # seconds
    tick = tick % (2 * math.pi)

    cross_position_x += speed

    if cross_position_x > SCREEN_WIDTH:
        cross_position_x = 0
        cross_position_y += speed

    if cross_position_y > SCREEN_HEIGHT:
        cross_position_y = 0
        cross_position_x = 0

    calib_img = etk.gen_red_circle_on_white_bg(
        size=(SCREEN_WIDTH, SCREEN_HEIGHT),
        circle_radius=20,
        circle_position=(int(cross_position_x), int(cross_position_y)),
    )

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

            # fit ellipse to left eye points
            left_eye_ellipse, left_eye_points = etk.fit_ellipse_to_eye(
                face_landmarks.landmark, LEFT_EYE_LANDMARKS, h, w
            )
            right_eye_ellipse, right_eye_points = etk.fit_ellipse_to_eye(
                face_landmarks.landmark, RIGHT_EYE_LANDMARKS, h, w
            )

            if left_eye_ellipse is not None and right_eye_ellipse is not None:
                # Crée un masque pour les yeux
                left_eye_img, left_eye_mask = etk.extract_eye_region(
                    frame, left_eye_ellipse
                )
                right_eye_img, right_eye_mask = etk.extract_eye_region(
                    frame, right_eye_ellipse
                )
                # Combine les deux images des yeux
                eyes = cv2.bitwise_or(left_eye_img, right_eye_img)

                # Recadre autour des yeux
                x, y, w_box, h_box = cv2.boundingRect(
                    np.array(left_eye_points + right_eye_points)
                )
                eyes_cropped = eyes[y : y + h_box, x : x + w_box]
                left_eye_img = left_eye_img[y : y + h_box, x : x + w_box]
                right_eye_img = right_eye_img[y : y + h_box, x : x + w_box]

                # Extraie les pupilles
                left_pupil = etk.extract_pupil3(
                    left_eye_img, left_eye_mask[y : y + h_box, x : x + w_box]
                )
                right_pupil = etk.extract_pupil3(
                    right_eye_img, right_eye_mask[y : y + h_box, x : x + w_box]
                )

                # get the centroid of each pupil
                M_left = cv2.moments(left_pupil)
                M_right = cv2.moments(right_pupil)

                left_bufferX.add(
                    int(M_left["m10"] / M_left["m00"]) if M_left["m00"] != 0 else 0
                )
                left_bufferY.add(
                    int(M_left["m01"] / M_left["m00"]) if M_left["m00"] != 0 else 0
                )

                right_bufferX.add(
                    int(M_right["m10"] / M_right["m00"]) if M_right["m00"] != 0 else 0
                )
                right_bufferY.add(
                    int(M_right["m01"] / M_right["m00"]) if M_right["m00"] != 0 else 0
                )

                # draw the centroid on the original frame

                cX_left = left_bufferX.get_median() + x
                cY_left = left_bufferY.get_median() + y
                cv2.circle(frame, (cX_left, cY_left), 2, (0, 255, 255), -1)

                cX_right = right_bufferX.get_median() + x
                cY_right = right_bufferY.get_median() + y
                cv2.circle(frame, (cX_right, cY_right), 2, (0, 255, 255), -1)

                # compute the centroid of each eye
                M_left_eye = cv2.moments(left_eye_mask[y : y + h_box, x : x + w_box])
                M_right_eye = cv2.moments(right_eye_mask[y : y + h_box, x : x + w_box])

                # draw the centroid on the original frame
                if M_left_eye["m00"] != 0:
                    cX_left_eye = int(M_left_eye["m10"] / M_left_eye["m00"]) + x
                    cY_left_eye = int(M_left_eye["m01"] / M_left_eye["m00"]) + y
                    cv2.circle(frame, (cX_left_eye, cY_left_eye), 2, (255, 255, 0), -1)
                if M_right_eye["m00"] != 0:
                    cX_right_eye = int(M_right_eye["m10"] / M_right_eye["m00"]) + x
                    cY_right_eye = int(M_right_eye["m01"] / M_right_eye["m00"]) + y
                    cv2.circle(
                        frame, (cX_right_eye, cY_right_eye), 2, (255, 255, 0), -1
                    )

        for i, (yaw, pitch, roll) in enumerate(face_orientation):
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

            # cv2.putText(frame, f"left pupil", (cX_left, cY_left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1 )
            # cv2.putText(frame, f"right pupil", (cX_right, cY_right), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1 )
            # cv2.putText(frame, f"left eye", (cX_left_eye, cY_left_eye), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1 )
            # cv2.putText(frame, f"right eye", (cX_right_eye, cY_right_eye), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1 )

            # save the data in the csv file
            noseX = face_landmarks.landmark[NOSE_LANDMARK].x
            noseY = face_landmarks.landmark[NOSE_LANDMARK].y
            leftEyeX = cX_left_eye / w
            leftEyeY = cY_left_eye / h
            rightEyeX = cX_right_eye / w
            rightEyeY = cY_right_eye / h
            leftEyeOuterX = face_landmarks.landmark[LEFT_EYE_OUTER].x
            leftEyeOuterY = face_landmarks.landmark[LEFT_EYE_OUTER].y
            rightEyeOuterX = face_landmarks.landmark[RIGHT_EYE_OUTER].x
            rightEyeOuterY = face_landmarks.landmark[RIGHT_EYE_OUTER].y
            leftEyeInnerX = face_landmarks.landmark[LEFT_EYE_INNER].x
            leftEyeInnerY = face_landmarks.landmark[LEFT_EYE_INNER].y
            rightEyeInnerX = face_landmarks.landmark[RIGHT_EYE_INNER].x
            rightEyeInnerY = face_landmarks.landmark[RIGHT_EYE_INNER].y
            leftEyeTopX = face_landmarks.landmark[LEFT_EYE_TOP].x
            leftEyeTopY = face_landmarks.landmark[LEFT_EYE_TOP].y
            rightEyeTopX = face_landmarks.landmark[RIGHT_EYE_TOP].x
            rightEyeTopY = face_landmarks.landmark[RIGHT_EYE_TOP].y
            leftEyeBottomX = face_landmarks.landmark[LEFT_EYE_BOTTOM].x
            leftEyeBottomY = face_landmarks.landmark[LEFT_EYE_BOTTOM].y
            rightEyeBottomX = face_landmarks.landmark[RIGHT_EYE_BOTTOM].x
            rightEyeBottomY = face_landmarks.landmark[RIGHT_EYE_BOTTOM].y

            print("Z de la face : ", face_landmarks.landmark[RIGHT_EYE_BOTTOM].z)

            # draw on the frame the landmarks used
            cv2.circle(
                frame,
                (int(leftEyeBottomX * w), int(leftEyeBottomY * h)),
                2,
                (255, 0, 0),
                -1,
            )
            cv2.circle(
                frame,
                (int(rightEyeBottomX * w), int(rightEyeBottomY * h)),
                2,
                (0, 0, 255),
                -1,
            )

            # compute the pupil position in the same frame of reference as the eye position
            leftPupilX = cX_left / w
            leftPupilY = cY_left / h
            rightPupilX = cX_right / w
            rightPupilY = cY_right / h
            csv_writer.writerow(
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
                    (cross_position_x) / SCREEN_WIDTH,
                    (cross_position_y) / SCREEN_HEIGHT,
                ]
            )

        # Convertir les landmarks de l'iris en coordonnées pixel
        left_iris_pts = np.array([(int(face_landmarks.landmark[i].x * w),
                                    int(face_landmarks.landmark[i].y * h)) for i in LEFT_IRIS_LANDMARKS])
        right_iris_pts = np.array([(int(face_landmarks.landmark[i].x * w),
                                    int(face_landmarks.landmark[i].y * h)) for i in RIGHT_IRIS_LANDMARKS])

        # Calculer le centre de chaque iris
        left_center = tuple(left_iris_pts.mean(axis=0).astype(int))
        right_center = tuple(right_iris_pts.mean(axis=0).astype(int))

        # Dessiner le contour de l’iris
        cv2.polylines(frame, [left_iris_pts], isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.polylines(frame, [right_iris_pts], isClosed=True, color=(0, 255, 0), thickness=1)

        # Dessiner le centre
        cv2.circle(frame, left_center, 3, (0, 0, 255), -1)
        cv2.circle(frame, right_center, 3, (0, 0, 255), -1)


        # Agrandit l'image 4x
        frame_big = cv2.resize(
            frame, (0, 0), fx=1.6, fy=1.6, interpolation=cv2.INTER_NEAREST
        )
        eye_cropped_big = cv2.resize(
            eyes_cropped, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST
        )
        eye_cropped_big_without = eye_cropped_big.copy()
        eye_cropped_big_without[:, :, 0] = 0  # Remove blue channel
        # eye_cropped_big_without[:, :, 1] = 0  # Remove green channel
        # eye_cropped_big_without[:, :, 2] = 0  # Remove red channel


        
        # show the result
        cv2.imshow("Eye Tracking", frame_big)
        cv2.imshow("Calibration", calib_img)
        # cv2.imshow("Extracted Eyes", eye_cropped_big if 'eye_cropped_big' in locals() else np.zeros((100, 100, 3), dtype=np.uint8))
        # cv2.imshow("Extracted Eyes Red", eye_cropped_big_without if 'eye_cropped_big_without' in locals() else np.zeros((100, 100, 3), dtype=np.uint8))

    # record video from webcam
    out.write(frame)
    # print("frame size :", frame.shape)

    if cv2.waitKey(1) & 0xFF == 27:  # Échap pour quitter
        break


cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
