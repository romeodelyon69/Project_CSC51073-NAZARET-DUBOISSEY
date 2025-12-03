import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy
import renderer3D as r3d

NOSE_LANDMARKS = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

LEFT_PUPIL = 473
RIGHT_PUPIL = 468

EYE_RADIUS_MM = 12  # Approximate eyeball radius in mm

class face:
    def __init__(self, frame, img_dims, cam_matrix=None):
        self.img_width = img_dims["width"]
        self.img_height = img_dims["height"]
        self.cam_matrix = cam_matrix
        self.landmarks = None
        self.ladnmarks_3D = None

        self.orientation_matrix = None
        self.face_center_px = None
        self.face_center_3D = None

        self.left_pupil_px = None
        self.right_pupil_px = None
        self.left_pupil_3D = None
        self.right_pupil_3D = None

        self.left_eyeball_ref_px = None
        self.right_eyeball_ref_px = None
        self.left_eyeball_px = None
        self.right_eyeball_px = None
        self.left_eyeball_3D = None
        self.right_eyeball_3D = None
        self.is_eyeball_ref_saved = False

        self.left_gaze_vector_3D = None
        self.right_gaze_vector_3D = None

        self.nose_keypoints = None
        self.translation = None
        self.scale_px_to_mm = None
        self.ref_scale_mm_to_px = None

        self.left_gaze_3D_pos_on_screen = None
        self.right_gaze_3D_pos_on_screen = None
        self.average_gaze_3D_pos_on_screen = None

        self.R_ref_nose = [None]  # Reference rotation matrix for nose stabilization
        
        self.renderer = r3d.Renderer3D(width=800, height=800)


    def compute_face_orientation(self, frame, indices, color=(0, 255, 0), size=80):
        # Extract 3D positions of selected landmarks
        points_3d = np.array([
            [self.landmarks[i].x * self.img_width, self.landmarks[i].y * self.img_height, self.landmarks[i].z * self.img_width]
            for i in indices
        ])

        # Compute the average position as the center of this substructure
        center = np.mean(points_3d, axis=0)

        # PCA-based orientation: Compute eigenvectors of the covariance matrix
        centered = points_3d - center
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvecs = eigvecs[:, np.argsort(-eigvals)]  # Sort by descending eigenvalue (major axes)

        # Ensure the orientation matrix is right-handed
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 2] *= -1

        # Convert to Euler angles and re-construct rotation matrix (optional but clarifies the transform)
        r = Rscipy.from_matrix(eigvecs)
        roll, pitch, yaw = r.as_euler('zyx', degrees=False)
        yaw *= 1
        roll *= 1
        R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()
        self.orientation = {"yaw": yaw, "pitch": pitch, "roll": roll}

        # === Stabilize rotation with reference matrix to avoid flipping during eigenvector sign change ===
        if self.R_ref_nose[0] is None:
            self.R_ref_nose[0] = R_final.copy()
        else:
            R_ref = self.R_ref_nose[0]
            for i in range(3):
                if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                    R_final[:, i] *= -1
        
        self.orientation_matrix = R_final
        self.face_center_px = center
        self.nose_keypoints = points_3d

    def compute_pupils(self):
        lm_left = self.landmarks[LEFT_PUPIL]
        lm_right = self.landmarks[RIGHT_PUPIL]

        x_left, y_left, z_left = lm_left.x * self.img_width, lm_left.y * self.img_height, lm_left.z * self.img_width
        x_right, y_right, z_right = lm_right.x * self.img_width, lm_right.y * self.img_height, lm_right.z * self.img_width

        self.left_pupil_px = np.array([x_left, y_left, z_left])
        self.right_pupil_px = np.array([x_right, y_right, z_right])

    def compute_center_3D_pos(self):
        center_x_px = self.face_center_px[0] 
        center_y_px = self.face_center_px[1]

        # 1. Constantes
        REAL_FACE_WIDTH = 140  # mm (Largeur bizygomatique moyenne)
        IDX_LEFT_SIDE = 454     # Point extrême gauche du visage
        IDX_RIGHT_SIDE = 234    # Point extrême droit du visage

        # 2. Récupérer les coordonnées en pixels
        lm_left = self.landmarks[IDX_LEFT_SIDE]
        lm_right = self.landmarks[IDX_RIGHT_SIDE]

        x_left, y_left, z_left = lm_left.x * self.img_width, lm_left.y * self.img_height, lm_left.z * self.img_width
        x_right, y_right, z_right = lm_right.x * self.img_width, lm_right.y * self.img_height, lm_right.z * self.img_width

        # 3. Calculer la distance apparente en pixels sur l'image
        pixel_dist = np.sqrt((x_left - x_right)**2 + (y_left - y_right)**2 + (z_left - z_right)**2)

        if pixel_dist < 1.0: 
            return None # Évite la division par zéro si erreur de détection

        # 4. Paramètres intrinsèques (issus de ta calibration)
        # fx est généralement en [0,0] et fy en [1,1] dans la matrice caméra
        fx = self.cam_matrix[0, 0] 
        cx = self.cam_matrix[0, 2] # Centre optique X
        cy = self.cam_matrix[1, 2] # Centre optique Y

        Z_mm = (fx * REAL_FACE_WIDTH) / pixel_dist

        X_mm = (center_x_px - cx) * Z_mm / fx
        Y_mm = (center_y_px - cy) * Z_mm / self.cam_matrix[1, 1]

        self.face_center_3D = np.array([X_mm, Y_mm, Z_mm])

    def compute_scale_px_to_mm(self):
        REAL_FACE_WIDTH = 140  # mm (Largeur bizygomatique moyenne)
        IDX_LEFT_SIDE = 454     # Point extrême gauche du visage
        IDX_RIGHT_SIDE = 234    # Point extrême droit du visage

        # 2. Récupérer les coordonnées en pixels
        lm_left = self.landmarks[IDX_LEFT_SIDE]
        lm_right = self.landmarks[IDX_RIGHT_SIDE]

        x_left, y_left, z_left = lm_left.x * self.img_width, lm_left.y * self.img_height, lm_left.z * self.img_width
        x_right, y_right, z_right = lm_right.x * self.img_width, lm_right.y * self.img_height, lm_right.z * self.img_width

        # 3. Calculer la distance apparente en pixels sur l'image
        pixel_dist = np.sqrt((x_left - x_right)**2 + (y_left - y_right)**2 + (z_left - z_right)**2)

        self.scale_px_to_mm = REAL_FACE_WIDTH / pixel_dist

    def save_eyeball_reference(self):
        self.compute_scale_px_to_mm()
        self.left_eyeball_ref_px = self.orientation_matrix.T @ (self.left_pupil_px  - self.face_center_px)
        self.right_eyeball_ref_px = self.orientation_matrix.T @ (self.right_pupil_px  - self.face_center_px)

        self.ref_scale_mm_to_px = 1/self.scale_px_to_mm

        camera_dir_world = np.array([0, 0, 1])
        camera_dir_local = self.orientation_matrix.T @ camera_dir_world
        print("camera dir local :", camera_dir_local)
        print("ref scale mm to px :", self.ref_scale_mm_to_px)
        print("eye size in mm :", EYE_RADIUS_MM)
        self.left_eyeball_ref_px += self.ref_scale_mm_to_px * EYE_RADIUS_MM * camera_dir_local
        self.right_eyeball_ref_px += self.ref_scale_mm_to_px * EYE_RADIUS_MM * camera_dir_local
        print("eye size in pixels :", self.scale_px_to_mm * EYE_RADIUS_MM)

        self.is_eyeball_ref_saved = True

    def compute_eyeball_positions(self):
        scale_factor = (1/self.scale_px_to_mm) / self.ref_scale_mm_to_px if self.ref_scale_mm_to_px is not None else 1.0
        if(self.left_eyeball_ref_px is not None and self.right_eyeball_ref_px is not None):
            self.left_eyeball_px = self.face_center_px + self.orientation_matrix @ (self.left_eyeball_ref_px * scale_factor)
            self.right_eyeball_px = self.face_center_px + self.orientation_matrix @ (self.right_eyeball_ref_px * scale_factor)

    def compute_gaze_vectors(self):
        if self.is_eyeball_ref_saved:
            self.left_gaze_vector_3D = self.left_pupil_3D - self.left_eyeball_3D
            self.right_gaze_vector_3D = self.right_pupil_3D - self.right_eyeball_3D

    def normalize_mediapipe_landmarks(self):
        self.compute_pupils()
        normalized_landmarks = []
        normalized_center = self.face_center_px * self.scale_px_to_mm
        for lm in self.landmarks:
            x_mm = lm.x * self.img_width * self.scale_px_to_mm
            y_mm = lm.y * self.img_height * self.scale_px_to_mm
            z_mm = lm.z * self.img_width * self.scale_px_to_mm
            land_mark_3D = np.array([x_mm, y_mm, z_mm]) - normalized_center + self.face_center_3D
            normalized_landmarks.append(land_mark_3D)

        self.landmarks_3D = normalized_landmarks

        self.left_pupil_3D = self.left_pupil_px * self.scale_px_to_mm - normalized_center + self.face_center_3D
        self.right_pupil_3D = self.right_pupil_px * self.scale_px_to_mm - normalized_center + self.face_center_3D

        if self.is_eyeball_ref_saved:
            self.left_eyeball_3D = self.left_eyeball_px * self.scale_px_to_mm - normalized_center + self.face_center_3D
            self.right_eyeball_3D = self.right_eyeball_px * self.scale_px_to_mm - normalized_center + self.face_center_3D

    def compute_gaze_intersection_on_screen(self, screen_z_mm=0):
        if not self.is_eyeball_ref_saved:
            return None, None

        def compute_intersection(eyeball_3D, gaze_vector_3D):
            t = (screen_z_mm - eyeball_3D[2]) / gaze_vector_3D[2]
            intersection_point = eyeball_3D + t * gaze_vector_3D
            return intersection_point

        left_intersection = compute_intersection(self.left_eyeball_3D, self.left_gaze_vector_3D)
        right_intersection = compute_intersection(self.right_eyeball_3D, self.right_gaze_vector_3D)

        self.left_gaze_3D_pos_on_screen = left_intersection
        self.right_gaze_3D_pos_on_screen = right_intersection
        self.average_gaze_3D_pos_on_screen = (left_intersection + right_intersection) / 2


    def update(self):
        self.compute_face_orientation(
            None,
            indices=NOSE_LANDMARKS
        )
        self.compute_scale_px_to_mm()
        self.compute_center_3D_pos()
        if self.is_eyeball_ref_saved:
            self.compute_eyeball_positions()
        self.normalize_mediapipe_landmarks()
        if self.is_eyeball_ref_saved:
            self.compute_gaze_vectors()
            self.compute_gaze_intersection_on_screen()

    def draw_world_landmarks(self):
        self.renderer.clear()

        # Draw landmarks
        for lm_3D in self.landmarks_3D:
            self.renderer.draw_point(lm_3D[0], lm_3D[1], lm_3D[2], color=(255, 255, 255), size=1)

        self.renderer.draw_point(
            self.left_pupil_3D[0], self.left_pupil_3D[1], self.left_pupil_3D[2],
            color=(255, 255, 0), size=5)
        self.renderer.draw_point(
            self.right_pupil_3D[0], self.right_pupil_3D[1], self.right_pupil_3D[2],
            color=(0, 255, 255), size=5)
        
        if self.is_eyeball_ref_saved:
            self.renderer.draw_point(
                self.left_eyeball_3D[0], self.left_eyeball_3D[1], self.left_eyeball_3D[2],
                color=(255, 0, 0), size=5)
            self.renderer.draw_point(
                self.right_eyeball_3D[0], self.right_eyeball_3D[1], self.right_eyeball_3D[2],
                color=(0, 0, 255), size=5)
            self.renderer.draw_vector(
                self.left_eyeball_3D, self.left_eyeball_3D + self.left_gaze_vector_3D * 3, color=(255, 255, 0), thickness=2)
            self.renderer.draw_vector(
                self.right_eyeball_3D, self.right_eyeball_3D + self.right_gaze_vector_3D * 3, color=(0, 255, 255), thickness=2)
            self.renderer.draw_point(
                self.left_gaze_3D_pos_on_screen[0], self.left_gaze_3D_pos_on_screen[1], self.left_gaze_3D_pos_on_screen[2],
                color=(255, 255, 0), size=3)
            self.renderer.draw_point(
                self.right_gaze_3D_pos_on_screen[0], self.right_gaze_3D_pos_on_screen[1], self.right_gaze_3D_pos_on_screen[2],
                color=(0, 255, 255), size=3)
            self.renderer.draw_point(
                self.average_gaze_3D_pos_on_screen[0], self.average_gaze_3D_pos_on_screen[1], self.average_gaze_3D_pos_on_screen[2],
                color=(255, 0, 255), size=6)

        self.renderer.write_text(
            f"Face Center 3D: X={self.face_center_3D[0]:.1f}mm, Y={self.face_center_3D[1]:.1f}mm, Z={self.face_center_3D[2]:.1f}mm",
            position=(-250,50,0),
            color=(255, 255, 255)
        )

        self.renderer.write_text(
            f"Face center mediapipe 3D (px) : X={self.face_center_px[0]:.1f}px, Y={self.face_center_px[1]:.1f}px, Z={self.face_center_px[2]:.1f}px",
            position=(-250,30,0),
            color=(255, 255, 255)
        )

        # Draw face orientation axes
        origin = self.face_center_3D
        x_axis = origin + self.orientation_matrix[:, 0] * 50  # 50 mm length
        y_axis = origin + self.orientation_matrix[:, 1] * 50
        z_axis = origin + self.orientation_matrix[:, 2] * 50

        self.renderer.draw_vector(origin, x_axis, color=(255, 0, 0), thickness=2)  # X axis in red
        self.renderer.draw_vector(origin, y_axis, color=(0, 255, 0), thickness=2)  # Y axis in green
        self.renderer.draw_vector(origin, z_axis, color=(0, 0, 255), thickness=2)  # Z axis in blue

        # Render the scene
        self.renderer.render(center= np.array([0,0,0]), show_axes=True)
        
