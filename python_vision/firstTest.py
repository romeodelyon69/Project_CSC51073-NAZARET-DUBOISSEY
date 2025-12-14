import cv2
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import numpy as np
import eyeTracker.eyeToolKit as etk
import math


import eyeTracker.Nlib as Nlib
from morse_decoder.pupil_decoder import PupilDecoder

from landmarks import (
    CHIN_LANDMARK,
    NOSE_LANDMARK,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    LEFT_MOUTH,
    RIGHT_MOUTH,
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    RIGHT_EYE_BOTTOM,
    LEFT_EYE_BOTTOM,
)
from utils import display_ypr

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True
)

cap = cv2.VideoCapture(0)

cross_position_x = 0
cross_position_y = 0
speed = 20  # pixels per frame


# record video from webcam in .MOV format
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (640, 480))


left_bufferX = Nlib.Buffer(9)
left_bufferY = Nlib.Buffer(9)
right_bufferX = Nlib.Buffer(9)
right_bufferY = Nlib.Buffer(9)
SCREEN_WIDTH = 1540
SCREEN_HEIGHT = 880

pupil_decoder = PupilDecoder()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    h, w, _ = frame.shape
    # print("h:", h, "w:", w)

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
        # print("number of faces:", len(results.multi_face_landmarks))
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
            pupil_decoder.add_entry(left_eye_points, right_eye_points)

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
                try:
                    left_pupil = etk.extract_pupil3(
                        left_eye_img, left_eye_mask[y : y + h_box, x : x + w_box]
                    )
                    right_pupil = etk.extract_pupil3(
                        right_eye_img, right_eye_mask[y : y + h_box, x : x + w_box]
                    )
                except Exception as e:
                    # print(f"Error extracting pupil: {e}")
                    left_pupil = None
                    right_pupil = None

                if left_pupil is not None and right_pupil is not None:
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
                        int(M_right["m10"] / M_right["m00"])
                        if M_right["m00"] != 0
                        else 0
                    )
                    right_bufferY.add(
                        int(M_right["m01"] / M_right["m00"])
                        if M_right["m00"] != 0
                        else 0
                    )

                    # draw the centroid on the original frame

                    cX_left = left_bufferX.get_median() + x
                    cY_left = left_bufferY.get_median() + y
                    cv2.circle(frame, (cX_left, cY_left), 2, (0, 255, 255), -1)

                    cX_right = right_bufferX.get_median() + x
                    cY_right = right_bufferY.get_median() + y
                    cv2.circle(frame, (cX_right, cY_right), 2, (0, 255, 255), -1)

                    # compute the centroid of each eye
                    M_left_eye = cv2.moments(
                        left_eye_mask[y : y + h_box, x : x + w_box]
                    )
                    M_right_eye = cv2.moments(
                        right_eye_mask[y : y + h_box, x : x + w_box]
                    )

                    # draw the centroid on the original frame
                    if M_left_eye["m00"] != 0:
                        cX_left_eye = int(M_left_eye["m10"] / M_left_eye["m00"]) + x
                        cY_left_eye = int(M_left_eye["m01"] / M_left_eye["m00"]) + y
                        cv2.circle(
                            frame, (cX_left_eye, cY_left_eye), 2, (255, 255, 0), -1
                        )
                    if M_right_eye["m00"] != 0:
                        cX_right_eye = int(M_right_eye["m10"] / M_right_eye["m00"]) + x
                        cY_right_eye = int(M_right_eye["m01"] / M_right_eye["m00"]) + y
                        cv2.circle(
                            frame, (cX_right_eye, cY_right_eye), 2, (255, 255, 0), -1
                        )

        for i, (yaw, pitch, roll) in enumerate(face_orientation):
            display_ypr(yaw, pitch, roll, frame, i, face_size)

            # cv2.putText(frame, f"left pupil", (cX_left, cY_left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1 )
            # cv2.putText(frame, f"right pupil", (cX_right, cY_right), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1 )
            # cv2.putText(frame, f"left eye", (cX_left_eye, cY_left_eye), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1 )
            # cv2.putText(frame, f"right eye", (cX_right_eye, cY_right_eye), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1 )


            # print("Z de la face : ", face_landmarks.landmark[RIGHT_EYE_BOTTOM].z)
            leftEyeBottomX = face_landmarks.landmark[LEFT_EYE_BOTTOM].x
            leftEyeBottomY = face_landmarks.landmark[LEFT_EYE_BOTTOM].y
            rightEyeBottomX = face_landmarks.landmark[RIGHT_EYE_BOTTOM].x
            rightEyeBottomY = face_landmarks.landmark[RIGHT_EYE_BOTTOM].y

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
cv2.destroyAllWindows()
