import cv2
import csv
import time
import sys
import numpy as np
from pathlib import Path
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import eyeToolKit as etk
from landmarks import LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS

# Configuration
IMAGE_CROP_SIZE = 50  # Fixed size for images around pupil (100x100 pixels)
DATA_BASE_DIR = Path("data/pupil_positions")

# Initialize MediaPipe Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True
)

# Generate unique ID (timestamp-based)
session_id = int(time.time())
session_dir = DATA_BASE_DIR / str(session_id)
images_dir = session_dir / "images"
session_dir.mkdir(parents=True, exist_ok=True)
images_dir.mkdir(parents=True, exist_ok=True)

# Create CSV file
csv_path = session_dir / "pupil_positions.csv"
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["id", "x", "y"])

print(f"Session ID: {session_id}")
print(f"Data will be saved to: {session_dir}")
print("Press ESC to stop recording...")

# Open webcam
cap = cv2.VideoCapture(0)
detection_id = 0  # Unique ID for each pupil detection

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results: Any = face_mesh.process(frame_rgb)

        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Fit ellipse to eye points
                left_eye_ellipse, left_eye_points = etk.fit_ellipse_to_eye(
                    face_landmarks.landmark, LEFT_EYE_LANDMARKS, h, w
                )
                right_eye_ellipse, right_eye_points = etk.fit_ellipse_to_eye(
                    face_landmarks.landmark, RIGHT_EYE_LANDMARKS, h, w
                )

                if left_eye_ellipse is not None and right_eye_ellipse is not None:
                    # Extract eye regions
                    left_eye_img, left_eye_mask = etk.extract_eye_region(
                        frame, left_eye_ellipse
                    )
                    right_eye_img, right_eye_mask = etk.extract_eye_region(
                        frame, right_eye_ellipse
                    )

                    # Crop around eyes
                    x, y, w_box, h_box = cv2.boundingRect(
                        np.array(left_eye_points + right_eye_points)
                    )
                    left_eye_img = left_eye_img[y : y + h_box, x : x + w_box]
                    right_eye_img = right_eye_img[y : y + h_box, x : x + w_box]
                    left_eye_mask = left_eye_mask[y : y + h_box, x : x + w_box]
                    right_eye_mask = right_eye_mask[y : y + h_box, x : x + w_box]

                    # Extract pupils
                    try:
                        left_pupil = etk.extract_pupil3(left_eye_img, left_eye_mask)
                        right_pupil = etk.extract_pupil3(right_eye_img, right_eye_mask)
                    except Exception as e:
                        print(f"Error extracting pupil: {e}")
                        left_pupil = None
                        right_pupil = None

                    # Process left pupil
                    if left_pupil is not None:
                        M_left = cv2.moments(left_pupil)
                        if M_left["m00"] != 0:
                            # Pupil position in cropped eye image
                            pupil_x_cropped = int(M_left["m10"] / M_left["m00"])
                            pupil_y_cropped = int(M_left["m01"] / M_left["m00"])

                            # Pupil position in original frame
                            pupil_x = pupil_x_cropped + x
                            pupil_y = pupil_y_cropped + y

                            # Save to CSV
                            csv_writer.writerow([detection_id, pupil_x, pupil_y])

                            # Extract fixed-size crop around pupil
                            half_size = IMAGE_CROP_SIZE // 2
                            x1 = max(0, pupil_x - half_size)
                            y1 = max(0, pupil_y - half_size)
                            x2 = min(w, pupil_x + half_size)
                            y2 = min(h, pupil_y + half_size)

                            # If crop goes out of bounds, pad with black
                            crop = frame[y1:y2, x1:x2].copy()
                            if (
                                crop.shape[0] < IMAGE_CROP_SIZE
                                or crop.shape[1] < IMAGE_CROP_SIZE
                            ):
                                padded = np.zeros(
                                    (IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3),
                                    dtype=np.uint8,
                                )
                                pad_y = (IMAGE_CROP_SIZE - crop.shape[0]) // 2
                                pad_x = (IMAGE_CROP_SIZE - crop.shape[1]) // 2
                                padded[
                                    pad_y : pad_y + crop.shape[0],
                                    pad_x : pad_x + crop.shape[1],
                                ] = crop
                                crop = padded

                            # Resize to exact size if needed
                            if (
                                crop.shape[0] != IMAGE_CROP_SIZE
                                or crop.shape[1] != IMAGE_CROP_SIZE
                            ):
                                crop = cv2.resize(
                                    crop, (IMAGE_CROP_SIZE, IMAGE_CROP_SIZE)
                                )

                            # Save image
                            image_path = images_dir / f"{detection_id:06d}.png"
                            cv2.imwrite(str(image_path), crop)

                            detection_id += 1

                            # Draw on frame for visualization
                            cv2.circle(frame, (pupil_x, pupil_y), 5, (0, 255, 0), -1)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Process right pupil
                    if right_pupil is not None:
                        M_right = cv2.moments(right_pupil)
                        if M_right["m00"] != 0:
                            # Pupil position in cropped eye image
                            pupil_x_cropped = int(M_right["m10"] / M_right["m00"])
                            pupil_y_cropped = int(M_right["m01"] / M_right["m00"])

                            # Pupil position in original frame
                            pupil_x = pupil_x_cropped + x
                            pupil_y = pupil_y_cropped + y

                            # Save to CSV
                            csv_writer.writerow([detection_id, pupil_x, pupil_y])

                            # Extract fixed-size crop around pupil
                            half_size = IMAGE_CROP_SIZE // 2
                            x1 = max(0, pupil_x - half_size)
                            y1 = max(0, pupil_y - half_size)
                            x2 = min(w, pupil_x + half_size)
                            y2 = min(h, pupil_y + half_size)

                            # If crop goes out of bounds, pad with black
                            crop = frame[y1:y2, x1:x2].copy()
                            if (
                                crop.shape[0] < IMAGE_CROP_SIZE
                                or crop.shape[1] < IMAGE_CROP_SIZE
                            ):
                                padded = np.zeros(
                                    (IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3),
                                    dtype=np.uint8,
                                )
                                pad_y = (IMAGE_CROP_SIZE - crop.shape[0]) // 2
                                pad_x = (IMAGE_CROP_SIZE - crop.shape[1]) // 2
                                padded[
                                    pad_y : pad_y + crop.shape[0],
                                    pad_x : pad_x + crop.shape[1],
                                ] = crop
                                crop = padded

                            # Resize to exact size if needed
                            if (
                                crop.shape[0] != IMAGE_CROP_SIZE
                                or crop.shape[1] != IMAGE_CROP_SIZE
                            ):
                                crop = cv2.resize(
                                    crop, (IMAGE_CROP_SIZE, IMAGE_CROP_SIZE)
                                )

                            # Save image
                            image_path = images_dir / f"{detection_id:06d}.png"
                            cv2.imwrite(str(image_path), crop)

                            detection_id += 1

                            # Draw on frame for visualization
                            cv2.circle(frame, (pupil_x, pupil_y), 5, (0, 255, 0), -1)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Display frame with info
        cv2.putText(
            frame,
            f"Session: {session_id} | Detections: {detection_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Pupil Position Recorder", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

finally:
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\nRecording stopped. Saved {detection_id} pupil detections.")
    print(f"CSV saved to: {csv_path}")
    print(f"Images saved to: {images_dir}")
