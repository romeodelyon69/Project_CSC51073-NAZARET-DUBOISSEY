import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy
import renderer3D as r3d

NOSE_LANDMARKS = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

class face:
    def __init__(self, frame, img_dims, cam_matrix=None):
        self.img_width = img_dims["width"]
        self.img_height = img_dims["height"]
        self.cam_matrix = cam_matrix
        self.landmarks = None
        self.ladnmarks_3D = None
        self.orientation_matrix = None
        self.face_center = None
        self.face_center_3D = None
        self.nose_keypoints = None
        self.translation = None
        self.scale_px_to_mm = None

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
        self.face_center = center
        self.nose_keypoints = points_3d

    def compute_center_3D_pos(self):
        center_x_px = self.face_center[0] 
        center_y_px = self.face_center[1]

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

    def normalize_mediapipe_landmarks(self):
        normalized_landmarks = []
        normalized_center = self.face_center * self.scale_px_to_mm
        for lm in self.landmarks:
            x_mm = lm.x * self.img_width * self.scale_px_to_mm
            y_mm = lm.y * self.img_height * self.scale_px_to_mm
            z_mm = lm.z * self.img_width * self.scale_px_to_mm
            land_mark_3D = np.array([x_mm, y_mm, z_mm]) - normalized_center + self.face_center_3D
            normalized_landmarks.append(land_mark_3D)

        self.landmarks_3D = normalized_landmarks

    def update(self):
        self.compute_face_orientation(
            None,
            indices=NOSE_LANDMARKS
        )
        self.compute_scale_px_to_mm()
        self.compute_center_3D_pos()
        self.normalize_mediapipe_landmarks()

    def draw_world_landmarks(self):
        self.renderer.clear()

        # Draw landmarks
        for lm_3D in self.landmarks_3D:
            self.renderer.draw_point(lm_3D[0], lm_3D[1], lm_3D[2], color=(255, 255, 255), size=1)

        self.renderer.write_text(
            f"Face Center 3D: X={self.face_center_3D[0]:.1f}mm, Y={self.face_center_3D[1]:.1f}mm, Z={self.face_center_3D[2]:.1f}mm",
            position=(-250,50,0),
            color=(255, 255, 255)
        )

        self.renderer.write_text(
            f"Face center mediapipe 3D (px) : X={self.face_center[0]:.1f}px, Y={self.face_center[1]:.1f}px, Z={self.face_center[2]:.1f}px",
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
        
