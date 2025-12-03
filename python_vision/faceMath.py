import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rscipy


DISTANCE_MENTON_FRONT_MM = 120.0 # Average distance from chin to front of face in mm

LEFT_IRIS_LANDMARKS = [474, 475, 477, 476]  # Left iris landmarks
RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]  # Right iris landmarks

LEFT_EYE_LMS = [133, 33, 159, 145]   # inner, outer, upper, lower eyelid
RIGHT_EYE_LMS = [362, 263, 386, 374]  # inner, outer, upper, lower eyelid

EYE_RADIUS_MM = 12.0   # mm
IRIS_RADIUS_MM = 6.0    # mm

LEFT_EYE_PUPIL = 468
RIGHT_EYE_PUPIL = 473

CHIN_LANDMARK = 152
GLABELLA_LANDMARK = 10

LEFT_EYE_OFFSET  = np.array([-20, -32, -15], dtype=float)   # mm
RIGHT_EYE_OFFSET = np.array([-20, +32, -15], dtype=float)

EyeRadius = 0.024  # Approximate diameter of the human eye in meters
def draw_wireframe_cube(frame, center, R, size=80):
    # Given a center and rotation matrix, draw a cube aligned to that orientation
    right = R[:, 0]
    up = -R[:, 1]
    forward = -R[:, 2]

    hw, hh, hd = size * 1, size * 1, size * 1

    def corner(x_sign, y_sign, z_sign):
        return (center +
                x_sign * hw * right +
                y_sign * hh * up +
                z_sign * hd * forward)

    # 8 corners of the cube
    corners = [corner(x, y, z) for x in [-1, 1] for y in [1, -1] for z in [-1, 1]]
    projected = [(int(pt[0]), int(pt[1])) for pt in corners]

    # Edges connecting the corners
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        cv2.line(frame, projected[i], projected[j], (255, 128, 0), 2)

def camera3D_to_pixel3D(image_params, X, Y, Z):
    """
    Convert 3D camera coordinates (X,Y,Z in mm)
    into pseudo '3D pixel coordinates' (u, v, d),
    where:
      u = pixel x
      v = pixel y
      d = depth (Z in mm)
    """

    fx = image_params["focal_length"]
    fy = image_params["focal_length"]
    cx = image_params["width"] / 2
    cy = image_params["height"] / 2

    # Projection pinhole camera model
    u = (X * fx / Z) + cx
    v = (Y * fy / Z) + cy
    d = Z  # keep depth as mm for accuracy

    return np.array([u, v, d], dtype=np.float32)

class face:
    def __init__(self, landmarks, image_params):
        self.landmarks = landmarks  # List of 3D landmark positions
        
        self.left_eye_vector = None
        self.right_eye_vector = None

        self.gaze_point = None

        self.left_eyeball= None
        self.right_eyeball = None

        self.left_eyeball_ref = None
        self.right_eyeball_ref = None
        self.scale_ref = None

        self.left_pupil = None
        self.right_pupil = None

        self.face_center = None
        self.nose_keypoints = None

        self.orientation = {}
        self.orientation_matrix = None

        self.R_ref_nose = [None]  # Reference rotation matrix for nose stabilization

        self.image_params = image_params  # Store image parameters (width, height, focal length, etc.)


    def compute_eye_vectors(self):
        # Compute eye direction vectors based on landmarks
        self.compute_mm_to_pixel_scale()
        self.compute_eyeball_positions()
        self.compute_pupil_positions()

        if self.left_eyeball is not None and self.right_eyeball is not None:
            self.left_eye_vector = self.left_pupil - self.left_eyeball
            self.right_eye_vector =  self.right_pupil - self.right_eyeball

    def compute_pupil_positions(self):
        self.left_pupil = np.array([int(self.landmarks[LEFT_EYE_PUPIL].x * self.image_params["width"]),
                           int(self.landmarks[LEFT_EYE_PUPIL].y * self.image_params["height"]),
                           int(self.landmarks[LEFT_EYE_PUPIL].z * self.image_params["width"])])

        self.right_pupil = np.array([int(self.landmarks[RIGHT_EYE_PUPIL].x * self.image_params["width"]),
                            int(self.landmarks[RIGHT_EYE_PUPIL].y * self.image_params["height"]),
                            int(self.landmarks[RIGHT_EYE_PUPIL].z * self.image_params["width"])])

    def compute_face_orientation(self, frame, indices, color=(0, 255, 0), size=80):
        # Extract 3D positions of selected landmarks
        points_3d = np.array([
            [self.landmarks[i].x * self.image_params["width"], self.landmarks[i].y * self.image_params["height"], self.landmarks[i].z * self.image_params["width"]]
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
    
    def save_eyeball_reference(self):
        self.left_eyeball_ref = self.orientation_matrix.T @ (self.left_pupil  - self.face_center)
        self.right_eyeball_ref = self.orientation_matrix.T @ (self.right_pupil  - self.face_center)

        self.scale_ref = self.compute_scale()
        self.compute_mm_to_pixel_scale()

        camera_dir_world = np.array([0, 0, 1])
        camera_dir_local = self.orientation_matrix.T @ camera_dir_world
        self.left_eyeball_ref += self.mm_to_pixel_scale * EYE_RADIUS_MM * camera_dir_local
        self.right_eyeball_ref += self.mm_to_pixel_scale * EYE_RADIUS_MM * camera_dir_local
        print("eye size in pixels :", self.mm_to_pixel_scale * EYE_RADIUS_MM)

        print("pos ref left eyeball :", self.left_eyeball_ref)
        print("pos ref right eyeball :", self.right_eyeball_ref)
        print("pos face center :", self.face_center)
        print("pos pupil left :", self.left_pupil)
        print("pos pupil right :", self.right_pupil)

    def compute_eyeball_positions(self):
        scale_factor = self.compute_scale() / self.scale_ref if self.scale_ref is not None else 1.0
        if(self.left_eyeball_ref is not None and self.right_eyeball_ref is not None):
            self.left_eyeball = self.face_center + self.orientation_matrix @ (self.left_eyeball_ref * scale_factor)
            self.right_eyeball = self.face_center + self.orientation_matrix @ (self.right_eyeball_ref * scale_factor)

    def compute_mm_to_pixel_scale(self):
        glabella = np.array([self.landmarks[GLABELLA_LANDMARK].x * self.image_params["width"],
                             self.landmarks[GLABELLA_LANDMARK].y * self.image_params["height"],
                             self.landmarks[GLABELLA_LANDMARK].z * self.image_params["width"]])
        chin = np.array([self.landmarks[CHIN_LANDMARK].x * self.image_params["width"],
                         self.landmarks[CHIN_LANDMARK].y * self.image_params["height"],
                         self.landmarks[CHIN_LANDMARK].z * self.image_params["width"]])
        pixel_dist = np.linalg.norm(glabella - chin)
        mm_dist = DISTANCE_MENTON_FRONT_MM
        self.mm_to_pixel_scale = pixel_dist / mm_dist

    def compute_scale(self):
        # Use average pairwise distance for robustness
        n = len(self.nose_keypoints)
        total = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(self.nose_keypoints[i] - self.nose_keypoints[j])
                total += dist
                count += 1
        return total / count if count > 0 else 1.0
    
    def compute_gaze_intersection(self):
        #compute the "intersection" of the two gaze vectors in 3D space i.e. the point that minimizes the distance to both gaze lines
        if self.left_eye_vector is None or self.right_eye_vector is None:
            return False
        p1 = self.left_eyeball
        d1 = self.left_eye_vector / np.linalg.norm(self.left_eye_vector)
        p2 = self.right_eyeball
        d2 = self.right_eye_vector / np.linalg.norm(self.right_eye_vector)

        # Compute closest points on each line
        v1 = p2 - p1
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, v1)
        e = np.dot(d2, v1)
        denom = a * c - b * b
        if abs(denom) < 1e-6:
            return None  # Lines are parallel
        t1 = (b * e - c * d) / denom
        t2 = (a * e - b * d) / denom
        closest_point1 = p1 + t1 * d1
        closest_point2 = p2 + t2 * d2
        intersection = (closest_point1 + closest_point2) / 2
        self.gaze_point = intersection
        return True

    def draw_face_landmarks(self, frame, indices, color=(0, 255, 0)):
        for i in indices:
            lm = self.landmarks[i]
            x, y = int(lm.x * self.image_params["width"]), int(lm.y * self.image_params["height"])
            cv2.circle(frame, (x, y), 1, color, -1)
        
        #draw face center
        if self.face_center is not None:
            cv2.circle(frame, (int(self.face_center[0]), int(self.face_center[1])), 3, (255, 0, 255), -1)

    def draw_pupil_positions(self, frame):
        # Draw pupil positions on the frame
        cv2.circle(frame, self.left_pupil[0:2], 2, (255, 0, 0), -1)
        cv2.circle(frame, self.right_pupil[0:2], 2, (0, 0, 255), -1)

    def draw_face_orientation(self, frame, size=80, color=(0, 255, 0)):
        if self.orientation_matrix is not None and self.face_center is not None:
            draw_wireframe_cube(frame, self.face_center, self.orientation_matrix, size)
            # Draw X (green), Y (blue), Z (red) axes
            axis_length = size * 1.2
            axis_dirs = [self.orientation_matrix[:, 0], -self.orientation_matrix[:, 1], -self.orientation_matrix[:, 2]]
            axis_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
            for i in range(3):
                end_pt = self.face_center + axis_dirs[i] * axis_length
                cv2.line(frame, (int(self.face_center[0]), int(self.face_center[1])), (int(end_pt[0]), int(end_pt[1])), axis_colors[i], 2)
            
            #write yaw pitch roll on frame
            cv2.putText(frame, f"Yaw: {np.degrees(self.orientation['yaw']):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Pitch: {np.degrees(self.orientation['pitch']):.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Roll: {np.degrees(self.orientation['roll']):.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def draw_eyeball_positions(self, frame):
        # Draw eyeball positions on the frame
        if self.left_eyeball is not None:
            cv2.circle(frame, (int(self.left_eyeball[0]), int(self.left_eyeball[1])), 3, (255, 255, 0), -1)
        if self.right_eyeball is not None:
            cv2.circle(frame, (int(self.right_eyeball[0]), int(self.right_eyeball[1])), 3, (0, 255, 255), -1)

        #ecrit à l'écran la postion du left eyeball et la position de la pupil left
        if self.left_eyeball is not None:
            cv2.putText(frame, f"Left Eyeball: ({self.left_eyeball[0]:.1f}, {self.left_eyeball[1]:.1f}, {self.left_eyeball[2]:.1f})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Left Pupil: ({self.left_pupil[0]:.1f}, {self.left_pupil[1]:.1f}, {self.left_pupil[2]:.1f})", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    
    def draw_debug_interface(self):
        """
        Display a 3D debug window showing:
        - face landmarks in 3D
        - eyeball positions
        - pupils
        - axes
        - rotatable 3D view (keys: A/D = rotate yaw, W/S = rotate pitch, Q/E = roll)
        """

        # === Init static attributes  ===
        if not hasattr(self, "_dbg_initialized"):
            self._dbg_initialized = True
            self._dbg_yaw = 0.0
            self._dbg_pitch = 0.0
            self._dbg_roll = 0.0
            self._dbg_scale = 0.8

        # === Collect 3D points to display ===
        pts = []
        pts_description = [] #(size, color)

        # Face landmarks in pseudo-3D (if available)
        for lm in self.landmarks:
            X = lm.x * self.image_params["width"]
            Y = lm.y * self.image_params["height"]
            Z = lm.z * self.image_params["width"]
            pts.append([X, Y, Z])
            pts_description.append((1, (255, 255, 255)))

        # Eyeballs
        if self.left_eyeball is not None:
            pts.append(self.left_eyeball)
            pts_description.append((5, (255, 255, 0)))
        if self.right_eyeball is not None:
            pts.append(self.right_eyeball)
            pts_description.append((5, (0, 255, 255)))

        # Pupils
        if self.left_pupil is not None:
            pts.append(self.left_pupil)
            pts_description.append((4, (255, 0, 0)))
        if self.right_pupil is not None:
            pts.append(self.right_pupil)
            pts_description.append((4, (0, 0, 255)))

        #gaze point
        if self.gaze_point is not None:
            pts.append(self.gaze_point)
            pts_description.append((6, (0, 255, 0)))

        pts = np.array(pts, dtype=float)

        # === Create 3D rotation matrix (yaw/pitch/roll in debug space) ===
        R_yaw = Rscipy.from_euler("y", self._dbg_yaw).as_matrix()
        R_pitch = Rscipy.from_euler("x", self._dbg_pitch).as_matrix()
        R_roll = Rscipy.from_euler("z", self._dbg_roll).as_matrix()
        Rview = R_roll @ R_pitch @ R_yaw

        # === Translate 3D points (center on face_center) ===
        if self.face_center is not None:
            pts_centered = pts - self.face_center
        else:
            pts_centered = pts.copy()

        # === Apply rotation ===
        pts_view = pts_centered @ Rview.T

        # === Simple orthographic projection ===
        # xz-plane used for display 
        win_w, win_h = 800, 800
        view = np.ones((win_h, win_w, 3), dtype=np.uint8) * 20

        proj = []
        for p in pts_view:
            x = int(p[0] * self._dbg_scale + win_w / 2)
            y = int(p[1] * self._dbg_scale + win_h / 2)
            proj.append((x, y))

        # === Draw points ===
        for (px, py), (size, color) in zip(proj, pts_description):
            cv2.circle(view, (px, py), size, color, -1)

        # === Draw eye direction vectors in 3D ===
        def draw_eye_vector(eyeball, eye_vec, color):
            if eyeball is None or eye_vec is None:
                return

            # Normalize and scale vector for visibility
            v = eye_vec.astype(float)
            norm = np.linalg.norm(v)
            if norm < 1e-6:
                return

            v = v / norm * 40.0  # length of arrow in pixels

            # Start = eyeball position
            start3d = eyeball.copy()
            end3d = eyeball + v

            # Transform into viewer coordinate system
            start3d_rel = start3d - self.face_center
            end3d_rel   = end3d   - self.face_center

            start3d_view = start3d_rel @ Rview.T
            end3d_view   = end3d_rel @ Rview.T

            start2d = start3d_view
            end2d   = end3d_view

            sx = int(start2d[0] * self._dbg_scale + win_w/2)
            sy = int(start2d[1] * self._dbg_scale + win_h/2)
            ex = int(end2d[0] * self._dbg_scale + win_w/2)
            ey = int(end2d[1] * self._dbg_scale + win_h/2)

            cv2.arrowedLine(view, (sx, sy), (ex, ey), color, 2, tipLength=0.2)

        # left = magenta, right = cyan
        draw_eye_vector(self.left_eyeball,  self.left_eye_vector,  (255,   0, 255))
        draw_eye_vector(self.right_eyeball, self.right_eye_vector, (  0, 255, 255))


        # === Draw axes ===
        axis_len = 50
        origin = np.array([0, 0, 0])
        axes = {
            "x": np.array([axis_len, 0, 0]),
            "y": np.array([0, axis_len, 0]),
            "z": np.array([0, 0, axis_len]),
        }

        for name, vec in axes.items():
            v_rot = vec @ Rview.T
            x1 = int(origin[0] * self._dbg_scale + win_w / 2)
            y1 = int(-origin[1] * self._dbg_scale + win_h / 2)
            x2 = int(v_rot[0] * self._dbg_scale + win_w / 2)
            y2 = int(-v_rot[1] * self._dbg_scale + win_h / 2)
            color = {"x": (0, 255, 0), "y": (255, 0, 0), "z": (0, 128, 255)}[name]
            cv2.line(view, (x1, y1), (x2, y2), color, 2)
            cv2.putText(view, name.upper(), (x2+5, y2+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # === Display window ===
        cv2.imshow("3D Debug View", view)

        # === Keyboard controls ===
        key = cv2.waitKey(1)
        if key == ord('q'): self._dbg_yaw -= 0.05
        if key == ord('d'): self._dbg_yaw += 0.05
        if key == ord('z'): self._dbg_pitch -= 0.05
        if key == ord('s'): self._dbg_pitch += 0.05
        if key == ord('a'): self._dbg_roll -= 0.05
        if key == ord('e'): self._dbg_roll += 0.05
        if key == ord('+'): self._dbg_scale *= 1.1
        if key == ord('-'): self._dbg_scale /= 1.1
 
    
        