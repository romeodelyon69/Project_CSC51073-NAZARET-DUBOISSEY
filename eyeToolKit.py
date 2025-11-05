import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


def fit_ellipse_to_eye(landmarks, eye_landmarks, h, w):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_landmarks]
    if len(points) >= 5:  # fitEllipse requires at least 5 points
        ellipse = cv2.fitEllipse(np.array(points))
        return ellipse, points
    return None, points


def extract_eye_region(frame, ellipse):
    eye_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.ellipse(eye_mask, ellipse, (255, 255, 255), -1)
    eye_region = cv2.bitwise_and(frame, frame, mask=eye_mask)
    return eye_region, eye_mask


def extract_pupil_old(eye_region, eye_mask):
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_eye], [0], eye_mask, [256], [0, 256])
    gray_eye = cv2.equalizeHist(gray_eye)
    _, pupil = cv2.threshold(gray_eye, 25, 255, cv2.THRESH_BINARY_INV)
    pupil = cv2.bitwise_and(pupil, pupil, mask=eye_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        pupil, connectivity=8
    )
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        pupil = np.zeros_like(pupil)
        pupil[labels == largest_label] = 255
    return pupil


def extract_pupil2(eye_region, eye_mask):
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_eye], [0], eye_mask, [256], [0, 256])
    gray_eye = cv2.equalizeHist(gray_eye)
    # extract the darkest 5% of pixels in the eye region
    threshold_value = np.percentile(gray_eye[eye_mask == 255], 5)
    _, pupil = cv2.threshold(gray_eye, threshold_value, 255, cv2.THRESH_BINARY_INV)
    pupil = cv2.bitwise_and(pupil, pupil, mask=eye_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        pupil, connectivity=8
    )
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        pupil = np.zeros_like(pupil)
        pupil[labels == largest_label] = 255
        return pupil


def extract_pupil3(eye_region, eye_mask):
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.equalizeHist(gray_eye)

    threshold_value = np.percentile(gray_eye[eye_mask == 255], 65)
    _, iris = cv2.threshold(gray_eye, threshold_value, 255, cv2.THRESH_BINARY_INV)
    iris = cv2.bitwise_and(iris, iris, mask=eye_mask)
    # close circle
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    iris = cv2.morphologyEx(iris, cv2.MORPH_CLOSE, kernel)
    # open circle
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    iris = cv2.morphologyEx(iris, cv2.MORPH_OPEN, kernel)

    # find the pupil as the darkest part of the iris
    iris_gray = cv2.bitwise_and(gray_eye, gray_eye, mask=iris)
    iris_gray = cv2.equalizeHist(iris_gray)

    # print("this is the iris gray : ")
    iris_gray_show = cv2.resize(
        iris_gray, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST
    )
    cv2.imshow("Iris Gray", iris_gray_show)

    threshold_value = np.percentile(iris_gray[iris == 255], 10)
    _, pupil = cv2.threshold(iris_gray, 25, 255, cv2.THRESH_BINARY_INV)

    pupil = cv2.bitwise_and(pupil, pupil, mask=iris)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        pupil, connectivity=8
    )

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        pupil = np.zeros_like(pupil)
        pupil[labels == largest_label] = 255

    return pupil


def get_face_orientation(
    landmarks,
    chin_idx,
    nose_idx,
    left_eye_idx,
    right_eye_idx,
    left_mouth_idx,
    right_mouth_idx,
):
    chin = np.array([landmarks[chin_idx].x, landmarks[chin_idx].y])
    nose = np.array([landmarks[nose_idx].x, landmarks[nose_idx].y])
    left_eye = np.array([landmarks[left_eye_idx].x, landmarks[left_eye_idx].y])
    right_eye = np.array([landmarks[right_eye_idx].x, landmarks[right_eye_idx].y])
    left_mouth = np.array([landmarks[left_mouth_idx].x, landmarks[left_mouth_idx].y])
    right_mouth = np.array([landmarks[right_mouth_idx].x, landmarks[right_mouth_idx].y])

    # compute yaw, pitch, roll
    eye_center = (left_eye + right_eye) / 2
    mouth_center = (left_mouth + right_mouth) / 2
    nose_to_chin = chin - nose
    eye_to_mouth = mouth_center - eye_center
    yaw = np.arctan2(eye_to_mouth[1], eye_to_mouth[0]) * 180 / np.pi
    pitch = np.arctan2(nose_to_chin[1], nose_to_chin[0]) * 180 / np.pi
    roll = (
        np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / np.pi
    )
    return yaw, pitch, roll


def get_face_size(landmarks, chin_idx, nose_idx, left_eye_idx, right_eye_idx):
    chin = np.array([landmarks[chin_idx].x, landmarks[chin_idx].y])
    nose = np.array([landmarks[nose_idx].x, landmarks[nose_idx].y])
    left_eye = np.array([landmarks[left_eye_idx].x, landmarks[left_eye_idx].y])
    right_eye = np.array([landmarks[right_eye_idx].x, landmarks[right_eye_idx].y])
    eye2eye_dist = np.linalg.norm(left_eye - right_eye)
    nose_to_chin_dist = np.linalg.norm(nose - chin)
    return np.sqrt(eye2eye_dist * nose_to_chin_dist)


def gen_red_circle_on_white_bg(
    size=(1920, 1080), circle_radius=20, circle_position=(0, 0)
):
    # print("position :", circle_position)
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    center = (circle_position[0], circle_position[1])
    cv2.circle(img, center, circle_radius, (0, 0, 255), -1)
    return img


def compute_gaze_physically(
    right_eye_yaw,
    left_eye_yaw,
    right_eye_pitch,
    left_eye_pitch,
    face_yaw,
    face_pitch,
    face_X,
    face_Y,
    face_Z,
):
    left_yaw = left_eye_yaw + face_yaw
    right_yaw = right_eye_yaw + face_yaw

    left_pitch = left_eye_pitch + face_pitch
    right_pitch = right_eye_pitch + face_pitch

    left_gaze_X = face_X + np.sin(np.radians(left_yaw)) * face_Z
    right_gaze_X = face_X + np.sin(np.radians(right_yaw)) * face_Z

    left_gaze_Y = face_Y + np.sin(np.radians(left_pitch)) * face_Z
    right_gaze_Y = face_Y + np.sin(np.radians(right_pitch)) * face_Z

    return left_gaze_X, right_gaze_X, left_gaze_Y, right_gaze_Y
