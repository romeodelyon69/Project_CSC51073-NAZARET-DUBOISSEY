import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy
import renderer3D as r3d
import Nlib
import eyeToolKit as etk

SCREEN_TRUE_WIDTH_MM = 345  # Largeur physique de l'écran en mm (exemple pour un écran 16" 16:9)
SCREEN_TRUE_HEIGHT_MM = 215  # Hauteur physique de l'écran en mm (exemple pour un écran 16" 16:9)

NOSE_LANDMARKS = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

RIGHT_EYE_LANDMARKS = [133, 27, 33, 23]

LEFT_EYE_LANDMARKS = [362, 257, 263, 253]

LEFT_PUPIL = 473
RIGHT_PUPIL = 468

EYE_RADIUS_MM = 10  # Approximate eyeball radius in mm

LEFT_CORNER_LANDMARK = 133
RIGHT_CORNER_LANDMARK = 362

CHIN_LANDMARK = 152
NOSE_LANDMARK = 1
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33
LEFT_EYE_INNER = 362
RIGHT_EYE_INNER = 133
LEFT_EYE_TOP = 159
RIGHT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 23
RIGHT_EYE_BOTTOM = 253
LEFT_MOUTH = 78
RIGHT_MOUTH = 308

STABILIZATION_FACTOR = 0.8

class face:
    def __init__(self, frame, img_dims, cam_matrix=None, dist_coeffs=None) -> None:
        self.img_width = img_dims["width"]
        self.img_height = img_dims["height"]
        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.landmarks = None
        self.landmarks_3D = None

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

#####################################################################################
#Features nécessaires pour l'implémentation de l'article Webcam-Based Visual Gaze Estimation
        self.left_eye_corner_px = None
        self.right_eye_corner_px = None
        self.left_eye_corner_3D = None
        self.right_eye_corner_3D = None
        self.left_indicator_mm = None
        self.right_indicator_mm = None
        self.left_indicator_calib = None
        self.right_indicator_calib = None
        self.calib_matrix_article_left = None
        self.calib_matrix_article_right = None

        self.screen_u_article = None
        self.screen_v_article = None

        self.compute_uv_methode = 0 #0: reconstruction 3D, 1: méthode article, 2: bout du nez       
#####################################################################################

        self.left_gaze_vector_3D = None
        self.right_gaze_vector_3D = None

        self.nose_keypoints = None
        self.translation = None
        self.scale_px_to_mm = None
        self.ref_scale_mm_to_px = None

        self.left_gaze_3D_pos_on_screen = None
        self.right_gaze_3D_pos_on_screen = None
        self.average_gaze_3D_pos_on_screen = None

        self.nose_vector_3D = None
        self.nose_intersection_point_3D_on_screen = None
        self.nose_uv_on_screen = None

        self.buffer_pos_on_screen = Nlib.VecBuffer(5)
        self.smoothed_gaze_on_screen = None
        self.screen_u = None
        self.screen_v = None


        self.screen_3D_calib_points = []
        self.screen_uv_calib_points = []
        self.is_screen_calibrated_3D_reconstruction = False
        self.is_screen_calibrated_article = False

        self.screen_calib_model = None

        self.R_ref_nose = [None]  # Reference rotation matrix for nose stabilization
        
        self.renderer = r3d.Renderer3D(width=800, height=800)
        self.face_is_centered = False


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

    def compute_nose_vector(self):
        #on considère que le vecteur du nez est l'axe Z de la matrice d'orientation que l'on fait pivoter autour de l'axe X
        #de 30 degrés pour corriger l'inclinaison naturelle du nez vers le bas
        angle_rad = np.deg2rad(-30)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
        self.nose_vector_3D = Rx @ self.orientation_matrix[:, 2]

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
        

        self.ref_scale_mm_to_px = 1/self.scale_px_to_mm

        camera_to_left_pupil = self.left_pupil_3D/np.linalg.norm(self.left_pupil_3D)
        camera_to_right_pupil = self.right_pupil_3D/np.linalg.norm(self.right_pupil_3D)
        print("ref scale mm to px :", self.ref_scale_mm_to_px)
        print("eye size in mm :", EYE_RADIUS_MM)
        left_Offset = self.ref_scale_mm_to_px * EYE_RADIUS_MM * camera_to_left_pupil
        right_Offset = self.ref_scale_mm_to_px * EYE_RADIUS_MM * camera_to_right_pupil
        print("eye size in pixels :", self.scale_px_to_mm * EYE_RADIUS_MM)

        self.left_eyeball_ref_px = self.orientation_matrix.T @ (self.left_pupil_px + left_Offset - self.face_center_px)
        self.right_eyeball_ref_px = self.orientation_matrix.T @ (self.right_pupil_px + right_Offset - self.face_center_px)

        self.is_eyeball_ref_saved = True

    def compute_eyeball_positions(self):
        scale_factor = (1/self.scale_px_to_mm) / self.ref_scale_mm_to_px if self.ref_scale_mm_to_px is not None else 1.0
        if(self.left_eyeball_ref_px is not None and self.right_eyeball_ref_px is not None):
            self.left_eyeball_px = self.face_center_px + self.orientation_matrix @ (self.left_eyeball_ref_px * scale_factor)
            self.right_eyeball_px = self.face_center_px + self.orientation_matrix @ (self.right_eyeball_ref_px * scale_factor)

    def compute_eyeball_position_from_landmarks(self):
        left_eye_3D = [self.landmarks_3D[i] for i in LEFT_EYE_LANDMARKS]
        right_eye_3D = [self.landmarks_3D[i] for i in RIGHT_EYE_LANDMARKS]
        left_eye_center_3D = np.mean(left_eye_3D, axis=0)
        right_eye_center_3D = np.mean(right_eye_3D, axis=0)

        def compute_normal_vector(points):
            res = np.array([0.0, 0.0, 0.0])
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    for k in range(j+1, len(points)):
                        v1 = points[j] - points[i]
                        v2 = points[k] - points[i]
                        normal = np.cross(v1, v2)
                        norm = np.linalg.norm(normal)
                        if norm > 1e-6:
                            res += normal / norm
            res /= np.linalg.norm(res)
            return res
            
        left_normal_vector = compute_normal_vector(left_eye_3D)
        right_normal_vector = compute_normal_vector(right_eye_3D)

        #si le vecteur normal pointe vers z>0 on l'inverse
        if left_normal_vector[2] > 0:
            left_normal_vector = -left_normal_vector
        if right_normal_vector[2] > 0:
            right_normal_vector = -right_normal_vector

        normal = (left_normal_vector + right_normal_vector) / 2
        normal /= np.linalg.norm(normal)

        self.left_eyeball_3D = left_eye_center_3D - EYE_RADIUS_MM * normal
        self.right_eyeball_3D = right_eye_center_3D - EYE_RADIUS_MM * normal

        #normal rotation de 30 degrés autour de l'axe X de l'axe Z de l'orientation
        angle_rad = np.deg2rad(-30)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
        normal = Rx @ self.orientation_matrix[:, 2]
        left_eye_center_3D = (self.landmarks_3D[LEFT_EYE_INNER] + self.landmarks_3D[LEFT_EYE_OUTER]) / 2
        right_eye_center_3D = (self.landmarks_3D[RIGHT_EYE_INNER] + self.landmarks_3D[RIGHT_EYE_OUTER]) / 2
        self.left_eyeball_3D = left_eye_center_3D - EYE_RADIUS_MM * normal
        self.right_eyeball_3D = right_eye_center_3D - EYE_RADIUS_MM * normal

    def compute_gaze_vectors(self):
        if self.is_eyeball_ref_saved:
            self.left_gaze_vector_3D = self.left_pupil_3D - self.left_eyeball_3D
            self.right_gaze_vector_3D = self.right_pupil_3D - self.right_eyeball_3D

            self.left_gaze_vector_3D /= np.linalg.norm(self.left_gaze_vector_3D)
            self.right_gaze_vector_3D /= np.linalg.norm(self.right_gaze_vector_3D)

            #amplification de la composante Y, 
            YAMP_FACTOR = 1.0
            self.left_gaze_vector_3D[1] = YAMP_FACTOR * self.left_gaze_vector_3D[1]
            self.right_gaze_vector_3D[1] = YAMP_FACTOR * self.right_gaze_vector_3D[1] 

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
        #self.buffer_pos_on_screen.add(self.right_gaze_3D_pos_on_screen)
        #self.smoothed_gaze_on_screen = self.buffer_pos_on_screen.get_mean()
        self.smoothed_gaze_on_screen = STABILIZATION_FACTOR * self.smoothed_gaze_on_screen + (1 - STABILIZATION_FACTOR) * self.average_gaze_3D_pos_on_screen if self.smoothed_gaze_on_screen is not None else self.average_gaze_3D_pos_on_screen

    def compute_nose_vector_intersection_on_screen(self, screen_z_mm=0):
        if self.nose_vector_3D is None or self.face_center_3D is None:
            return None

        t = (screen_z_mm - self.face_center_3D[2]) / self.nose_vector_3D[2]
        intersection_point = self.face_center_3D + t * self.nose_vector_3D
        self.nose_intersection_point_3D_on_screen = intersection_point
    
    def compute_nose_uv_on_screen(self):
        #Dans le repère caméra, l'écran à les coordonnées suivante Z=0 et :
        #top left : (X=+SCREEN_TRUE_WIDTH_MM/2, Y=0)
        #top right : (X=-SCREEN_TRUE_WIDTH_MM/2, Y=0)
        #bottom left : (X=+SCREEN_TRUE_WIDTH_MM/2, Y=+SCREEN_TRUE_HEIGHT_MM/)
        #bottom right : (X=-SCREEN_TRUE_WIDTH_MM/2, Y=+SCREEN_TRUE_HEIGHT_MM/2)
        #On calcule les coordonnées u,v normalisées entre 0 et 1
        if self.nose_intersection_point_3D_on_screen is None:
            return None
        Xg, Yg, Zg = self.nose_intersection_point_3D_on_screen
        u = 1/2 - Xg / SCREEN_TRUE_WIDTH_MM
        v = Yg / SCREEN_TRUE_HEIGHT_MM
        self.nose_uv_on_screen = STABILIZATION_FACTOR * self.nose_uv_on_screen + (1 - STABILIZATION_FACTOR) * np.array((u, v)) if self.nose_uv_on_screen is not None else np.array((u, v))

    def get_smoothed_gaze_on_screen(self):
        return self.smoothed_gaze_on_screen

    def compute_gaze_screen_coordinates_global(self):
        if not self.is_screen_calibrated_3D_reconstruction:
            return
        Xg, Yg, Zg = self.smoothed_gaze_on_screen
        
        # Produit scalaire : [Xg, Yg, 1] . [a1, a2, a3]
        u, v = self.map_3D_to_uv(Xg, Yg, Zg)
        self.screen_u = u
        self.screen_v = v

    def set_screen_calibration_data_3D_reconstruction(self, data, model_type='quadratic'):
        """
        data: dict with keys:
        - "screen_3D_calib_points": list of (X,Y,Z) or (X,Y) tuples
        - "screen_uv_calib_points": list of (u,v) tuples (normalized [0,1])
        model_type: 'affine', 'quadratic', or 'homography'
        """
        # store raw data (fixing possible naming inconsistencies)
        self.screen_3D_calib_points = data["screen_3D_calib_points"]
        self.screen_uv_calib_points = data["screen_uv_calib_points"]
        self.is_screen_calibrated_3D_reconstruction = True

        N = len(self.screen_3D_calib_points)
        if N < 3:
            # pas assez de points pour calibrer (au minimum 3 pour homographie, 3 pour quadratic/affine)
            return {"status": "not_enough_points", "num_points": N}

        # build arrays
        pts3 = np.asarray(self.screen_3D_calib_points, dtype=float)
        pts2 = np.asarray(self.screen_uv_calib_points, dtype=float)

        # handle if 3D supplied (X,Y,Z) but we only use X,Y; assume Z=0 plane
        if pts3.shape[1] >= 2:
            X = pts3[:, 0]
            Y = pts3[:, 1]
        else:
            raise ValueError("screen_3D_calib_points must have at least X and Y")

        u = pts2[:, 0]
        v = pts2[:, 1]

        result = {"model_type": model_type, "num_points": N}

        if model_type == 'affine':
            # design matrix: [X, Y, 1]
            A = np.column_stack([X, Y, np.ones(N)])
            # solve for u and v separately
            au, *_ = np.linalg.lstsq(A, u, rcond=None)
            av, *_ = np.linalg.lstsq(A, v, rcond=None)
            # store params
            self.screen_calib_model = {
                "type": "affine",
                "Au": au,  # shape (3,)
                "Av": av,  # shape (3,)
            }
            # compute predictions and RMSE
            pred_u = A.dot(au)
            pred_v = A.dot(av)
        elif model_type == 'quadratic':
            # design matrix: [X^2, X*Y, Y^2, X, Y, 1]
            A = np.column_stack([X**2, X*Y, Y**2, X, Y, np.ones(N)])
            cu, *_ = np.linalg.lstsq(A, u, rcond=None)
            cv, *_ = np.linalg.lstsq(A, v, rcond=None)
            self.screen_calib_model = {
                "type": "quadratic",
                "Cu": cu,  # shape (6,)
                "Cv": cv,
            }
            pred_u = A.dot(cu)
            pred_v = A.dot(cv)
        elif model_type == 'homography':
            # need at least 4 points for robust homography; with 3 points you can fit but degenerate.
            if N < 4:
                # still try, but warn in result
                warn = "homography_fitted_with_less_than_4_points"
            else:
                warn = None
            # build 2N x 9 matrix
            # For each correspondence (X,Y) -> (u,v) (we treat u,v as inhomogeneous coords)
            # Equation: [ -X -Y -1  0  0  0  uX uY u ] and [ 0 0 0 -X -Y -1  vX vY v ]
            M = []
            for xi, yi, ui, vi in zip(X, Y, u, v):
                M.append([-xi, -yi, -1, 0, 0, 0, ui*xi, ui*yi, ui])
                M.append([0, 0, 0, -xi, -yi, -1, vi*xi, vi*yi, vi])
            M = np.asarray(M, dtype=float)
            # SVD
            U, S, Vt = np.linalg.svd(M)
            # last column of V (or last row of Vt) is solution
            h = Vt[-1, :]
            H = h.reshape(3, 3)
            # normalize so H[2,2] = 1 if possible
            if abs(H[2, 2]) > 1e-12:
                H = H / H[2, 2]
            self.screen_calib_model = {
                "type": "homography",
                "H": H,
                "warn": warn
            }
            # compute predicted (inhomogeneous)
            denom = H[2,0]*X + H[2,1]*Y + H[2,2]
            pred_u = (H[0,0]*X + H[0,1]*Y + H[0,2]) / denom
            pred_v = (H[1,0]*X + H[1,1]*Y + H[1,2]) / denom
        else:
            raise ValueError("Unknown model_type: choose 'affine','quadratic' or 'homography'")

        # compute errors
        err_u = pred_u - u
        err_v = pred_v - v
        rmse_u = np.sqrt(np.mean(err_u**2))
        rmse_v = np.sqrt(np.mean(err_v**2))
        rmse = np.sqrt(np.mean(err_u**2 + err_v**2))

        result.update({
            "rmse_u": float(rmse_u),
            "rmse_v": float(rmse_v),
            "rmse": float(rmse),
        })
        if 'warn' in locals() and warn is not None:
            result['warning'] = warn

        # store residuals and original points if desired
        self.screen_calib_result = result

    # provide a prediction function accessible via self.map_3D_to_uv
    def map_3D_to_uv(self, x, y, z=None):
        x = float(x); y = float(y)
        if self.screen_calib_model["type"] == "affine":
            au = self.screen_calib_model["Au"]
            av = self.screen_calib_model["Av"]
            uu = au[0]*x + au[1]*y + au[2]
            vv = av[0]*x + av[1]*y + av[2]
            return uu, vv
        elif self.screen_calib_model["type"] == "quadratic":
            C = self.screen_calib_model
            Cu = C["Cu"]; Cv = C["Cv"]
            Arow = np.array([x**2, x*y, y**2, x, y, 1.0])
            uu = float(Arow.dot(Cu))
            vv = float(Arow.dot(Cv))
            return uu, vv
        else:  # homography
            H = self.screen_calib_model["H"]
            denom = H[2,0]*x + H[2,1]*y + H[2,2]
            uu = (H[0,0]*x + H[0,1]*y + H[0,2]) / denom
            vv = (H[1,0]*x + H[1,1]*y + H[1,2]) / denom
            return float(uu), float(vv)
        


    def set_screen_calibration_data_article(self, data, method='quadratic'):
        """
        Calibre le regard en utilisant soit une approche linéaire (affine), 
        soit une approche quadratique (polynôme de degré 2).
        
        Args:
            data: dict contenant les clés "screen_uv_calib_points", "left_indicators", "right_indicators"
            method: str, 'linear' ou 'quadratic'
        """
        # 1. Stockage des données brutes
        self.screen_uv_calib_points = data["screen_uv_calib_points"]
        self.left_indicator_calib = data["left_indicators"]   
        self.right_indicator_calib = data["right_indicators"] 
        
        # On sauvegarde la méthode choisie pour l'utiliser lors de l'inférence
        self.calibration_method_article = method

        try:
            # Conversion en numpy array
            X_left_raw = np.array(self.left_indicator_calib)   # Shape (N, 2)
            X_right_raw = np.array(self.right_indicator_calib) # Shape (N, 2)
            Y_screen = np.array(self.screen_uv_calib_points)   # Shape (N, 2)

            # Vérification
            if len(X_left_raw) != len(Y_screen) or len(X_right_raw) != len(Y_screen):
                print(f"Erreur dimensions: Left={len(X_left_raw)}, Right={len(X_right_raw)}, Screen={len(Y_screen)}")
                self.is_screen_calibrated_article = False
                return

            # --- 2. Préparation des Features (Matrice de Design) ---
            def get_design_matrix(vectors, mode):
                """Transforme les vecteurs [x, y] en features selon le mode."""
                x = vectors[:, 0]
                y = vectors[:, 1]
                
                if mode == 'quadratic':
                    # Formule: a*x^2 + b*y^2 + c*xy + d*x + e*y + f
                    # Features: [x^2, y^2, xy, x, y, 1]
                    return np.column_stack([x**2, y**2, x*y, x, y, np.ones_like(x)])
                else: # 'linear'
                    # Formule: a*x + b*y + c
                    # Features: [x, y, 1]
                    return np.column_stack([x, y, np.ones_like(x)])

            # Transformation des entrées
            X_left_features = get_design_matrix(X_left_raw, method)
            X_right_features = get_design_matrix(X_right_raw, method)

            # --- 3. Résolution (Least Squares) ---
            def compute_matrix(A, targets):
                # A * M = Y  => M = lstsq(A, Y)
                coeffs, _, _, _ = np.linalg.lstsq(A, targets, rcond=None)
                return coeffs

            if len(Y_screen) >= 6 if method == 'quadratic' else 2:
                self.calib_matrix_article_left = compute_matrix(X_left_features, Y_screen)
                self.calib_matrix_article_right = compute_matrix(X_right_features, Y_screen)
                
                self.is_screen_calibrated_article = True
                
                # --- 4. Calcul et affichage de l'erreur ---
                def compute_error(A, targets, matrix):
                    predictions = np.dot(A, matrix)
                    diff = predictions - targets
                    distances = np.linalg.norm(diff, axis=1)
                    return np.mean(distances)

                err_left = compute_error(X_left_features, Y_screen, self.calib_matrix_article_left)
                err_right = compute_error(X_right_features, Y_screen, self.calib_matrix_article_right)

                print(f"--- RÉSULTATS CALIBRATION ({method.upper()}) ---")
                print(f"Erreur Moyenne Oeil GAUCHE : {err_left:.4f} (UV)")
                print(f"Erreur Moyenne Oeil DROIT  : {err_right:.4f} (UV)")
                
                avg_err = (err_left + err_right) / 2
                if avg_err < 0.1: print(">> Qualité : BONNE")
                elif avg_err < 0.2: print(">> Qualité : MOYENNE")
                else: print(">> Qualité : MAUVAISE")

            else:
                print(f"Erreur : Pas assez de points pour une calibration {method}.")
                self.is_screen_calibrated_article = False

        except Exception as e:
            print(f"Exception critique durant la calibration : {e}")
            self.is_screen_calibrated_article = False
    

    def map_indicators_to_screen_article(self, method='quadratic'):
        """
        Effectue l'inférence en respectant la méthode de calibration (linéaire ou quadratique).
        Utilise self.left_indicator_px et self.right_indicator_px (vecteurs actuels).
        """
        if not getattr(self, "is_screen_calibrated_article", False):
            return None


        def transform_and_map(vector, matrix):
            if vector is None: return None
            vx, vy = vector
            
            # Construction du vecteur de features (Doit correspondre exactement à get_design_matrix)
            if method == 'quadratic':
                # [x^2, y^2, xy, x, y, 1]
                features = np.array([vx**2, vy**2, vx*vy, vx, vy, 1.0])
            else:
                # [x, y, 1]
                features = np.array([vx, vy, 1.0])
                
            # Produit scalaire
            return np.dot(features, matrix)

        # Calcul pour chaque oeil
        left_point = None
        right_point = None

        if self.left_indicator_mm is not None:
            left_point = transform_and_map(self.left_indicator_mm, self.calib_matrix_article_left)
            
        if self.right_indicator_mm is not None:
            right_point = transform_and_map(self.right_indicator_mm, self.calib_matrix_article_right)
        # Moyenne des deux yeux
        if left_point is not None and right_point is not None:
            res = (left_point + right_point) / 2
        elif left_point is not None:
            res = tuple(left_point)
        elif right_point is not None:
            res = tuple(right_point)
        
        self.screen_u_article, self.screen_v_article = res
        return res

    def compute_eye_corners(self):
        lm_left_outer = self.landmarks[LEFT_CORNER_LANDMARK]
        lm_right_outer = self.landmarks[RIGHT_CORNER_LANDMARK]

        x_left_outer, y_left_outer, z_left_outer = lm_left_outer.x * self.img_width, lm_left_outer.y * self.img_height, lm_left_outer.z * self.img_width
        x_right_outer, y_right_outer, z_right_outer = lm_right_outer.x * self.img_width, lm_right_outer.y * self.img_height, lm_right_outer.z * self.img_width

        self.left_eye_corner_px = np.array([x_left_outer, y_left_outer, z_left_outer])
        self.right_eye_corner_px = np.array([x_right_outer, y_right_outer, z_right_outer])

        #compute 3D positions
        self.left_eye_corner_3D = (self.left_eye_corner_px - self.face_center_px) * self.scale_px_to_mm + self.face_center_3D
        self.right_eye_corner_3D = (self.right_eye_corner_px - self.face_center_px) * self.scale_px_to_mm + self.face_center_3D

    def compute_indicator_positions_mm(self):
        self.compute_eye_corners()
        self.left_indicator_mm = (self.left_pupil_px - self.left_eye_corner_px)[0:2] * self.scale_px_to_mm        #à voir si on garde le z ou pas
        self.right_indicator_mm = (self.right_pupil_px - self.right_eye_corner_px)[0:2] * self.scale_px_to_mm     

    def get_indicator_positions_mm(self):
        return self.left_indicator_mm, self.right_indicator_mm

    def get_screen_coordinates(self):
        if self.compute_uv_methode == 0 and self.is_screen_calibrated_3D_reconstruction:
            return self.screen_u, self.screen_v
        if self.compute_uv_methode == 1 and self.is_screen_calibrated_article:
            return self.screen_u_article, self.screen_v_article
        if self.compute_uv_methode == 2:
            return self.nose_uv_on_screen
        return None, None
    
    def update(self):
        self.compute_face_orientation(
            None,
            indices=NOSE_LANDMARKS
        )
        self.compute_scale_px_to_mm()
        self.compute_center_3D_pos()
        self.compute_nose_vector()
        self.compute_nose_vector_intersection_on_screen()
        self.compute_nose_uv_on_screen()
        if self.is_eyeball_ref_saved:
            self.compute_eyeball_positions()
        self.normalize_mediapipe_landmarks()

        #override eyeball position with landmark-based estimation
        self.compute_eyeball_position_from_landmarks() #la construction est moins bonne mais plus stable

        self.compute_indicator_positions_mm()
        if self.is_eyeball_ref_saved:
            self.compute_gaze_vectors()
            self.compute_gaze_intersection_on_screen()
            self.compute_gaze_screen_coordinates_global()
        if self.is_screen_calibrated_article:
            self.map_indicators_to_screen_article()



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
        
        # self.renderer.draw_point(
        #     self.left_eye_corner_3D[0], self.left_eye_corner_3D[1], self.left_eye_corner_3D[2],
        #     color=(255, 0, 0), size=5)    #BLUE
        # self.renderer.draw_point(
        #     self.right_eye_corner_3D[0], self.right_eye_corner_3D[1], self.right_eye_corner_3D[2],
        #     color=(0, 0, 255), size=5)    #RED

        for landmark_id in LEFT_EYE_LANDMARKS:
            lm_3D = self.landmarks_3D[landmark_id]
            self.renderer.draw_point(lm_3D[0], lm_3D[1], lm_3D[2], color=(255, 0, 0), size=3)
        for landmark_id in RIGHT_EYE_LANDMARKS:
            lm_3D = self.landmarks_3D[landmark_id]
            self.renderer.draw_point(lm_3D[0], lm_3D[1], lm_3D[2], color=(0, 0, 255), size=3)

         # Draw eyeball and gaze vectors
        if self.is_eyeball_ref_saved:
            self.renderer.draw_point(
                self.left_eyeball_3D[0], self.left_eyeball_3D[1], self.left_eyeball_3D[2],
                color=(255, 0, 0), size=5)
            self.renderer.draw_point(
                self.right_eyeball_3D[0], self.right_eyeball_3D[1], self.right_eyeball_3D[2],
                color=(0, 0, 255), size=5)
            self.renderer.draw_vector(
                self.left_eyeball_3D, self.left_eyeball_3D + self.left_gaze_vector_3D * 60, color=(255, 255, 0), thickness=2)
            self.renderer.draw_vector(
                self.right_eyeball_3D, self.right_eyeball_3D + self.right_gaze_vector_3D * 60, color=(0, 255, 255), thickness=2)
            self.renderer.draw_point(
                self.left_gaze_3D_pos_on_screen[0], self.left_gaze_3D_pos_on_screen[1], self.left_gaze_3D_pos_on_screen[2],
                color=(255, 255, 0), size=3)
            self.renderer.draw_point(
                self.right_gaze_3D_pos_on_screen[0], self.right_gaze_3D_pos_on_screen[1], self.right_gaze_3D_pos_on_screen[2],
                color=(0, 255, 255), size=3)
            self.renderer.draw_point(
                self.average_gaze_3D_pos_on_screen[0], self.average_gaze_3D_pos_on_screen[1], self.average_gaze_3D_pos_on_screen[2],
                color=(255, 0, 255), size=6)
            self.renderer.draw_point(
                self.smoothed_gaze_on_screen[0], self.smoothed_gaze_on_screen[1], self.smoothed_gaze_on_screen[2],
                color=(0, 255, 0), size=8)
            
        if self.is_screen_calibrated_3D_reconstruction:
             # Draw 3D screen points avec un dégradé de couleur pour visualiser l'ordre des points et leur position
            for i, lm_3D in enumerate(self.screen_3D_calib_points):
                color = (0, 255 - int(i * (255 / len(self.screen_3D_calib_points))), int(i * (255 / len(self.screen_3D_calib_points))))
                self.renderer.draw_point(lm_3D[0], lm_3D[1], lm_3D[2], color=color, size=3)
            
            message = "Press 'c' to start calibration"

            self.renderer.write_text(
                    message,
                    position=(-250, 260, 0),
                    color=(255, 255, 255)
                )
            
        #dessine le vecteur du nez
        nose_end = self.face_center_3D + self.nose_vector_3D * 100  # 100 mm length
        self.renderer.draw_vector(
            self.face_center_3D, nose_end,
            color=(0, 165, 255), thickness=3)  # Orange color
        

        #dessine l'écran théorique dans le cadre de mon ordi i.e. 16 pouces avec la caméra au centre en haut
        screen_corners = [
            [-SCREEN_TRUE_WIDTH_MM/2, 0, 0],   # Top-left
            [SCREEN_TRUE_WIDTH_MM/2, 0, 0],    # Top-right
            [SCREEN_TRUE_WIDTH_MM/2, SCREEN_TRUE_HEIGHT_MM, 0],# Bottom-right
            [-SCREEN_TRUE_WIDTH_MM/2, SCREEN_TRUE_HEIGHT_MM, 0]# Bottom-left
        ]
        for i in range(len(screen_corners)):
            start = screen_corners[i]
            end = screen_corners[(i + 1) % len(screen_corners)]
            self.renderer.draw_vector(
                np.array(start), np.array(end),
                color=(165, 0, 255), thickness=1)
            
        for corner in screen_corners:
            self.renderer.draw_point(
                corner[0], corner[1], corner[2],
                color=(165, 0, 255), size=4)
        
        #dessine le point d'intersection du vecteur du nez avec l'écran
        if self.nose_intersection_point_3D_on_screen is not None:   
            self.renderer.draw_point(
                self.nose_intersection_point_3D_on_screen[0],
                self.nose_intersection_point_3D_on_screen[1],
                self.nose_intersection_point_3D_on_screen[2],
                color=(0, 165, 255), size=6)  # Orange color

        self.renderer.write_text(
            f"Face Center 3D: X={self.face_center_3D[0]:.1f}mm, Y={self.face_center_3D[1]:.1f}mm, Z={self.face_center_3D[2]:.1f}mm",
            position=(-250,200,0),
            color=(255, 255, 255)
        )

        self.renderer.write_text(
            f"Face center mediapipe 3D (px) : X={self.face_center_px[0]:.1f}px, Y={self.face_center_px[1]:.1f}px, Z={self.face_center_px[2]:.1f}px",
            position=(-250,230,0),
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
        if self.face_is_centered:
            self.renderer.render(center= self.face_center_3D, show_axes=True)
        else:
            self.renderer.render(center= np.array([0,0,0]), show_axes=True)
        
