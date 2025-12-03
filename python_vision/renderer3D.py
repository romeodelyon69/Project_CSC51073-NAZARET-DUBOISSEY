import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


class Renderer3D:
    def __init__(self, width=800, height=800):
        # Window params
        self.width = width
        self.height = height

        # View params
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.scale = 1.0

        # Storage for primitives
        self.points = []   # (x,y,z,size,color)
        self.vectors = []  # (start, end, color, thickness)
        self.text = []     # (text, position, color)

        # Background color
        self.bg_color = (20, 20, 20)

    # --------------------------------------------------------------
    #   METHODS TO ADD CONTENT
    # --------------------------------------------------------------
    def draw_point(self, x, y, z, color=(255,255,255), size=3):
        self.points.append((np.array([x, y, z], dtype=float), size, color))

    def draw_vector(self, start, end, color=(0,255,0), thickness=2):
        self.vectors.append((np.array(start, float), np.array(end, float), color, thickness))

    def write_text(self, text, position, color=(255,255,255), thickness=2, scale=0.6):
        self.text.append((text, position, color, thickness, scale))

    def clear(self):
        self.points = []
        self.vectors = []
        self.text = []

    # --------------------------------------------------------------
    #  PROJECTION 3D -> 2D
    # --------------------------------------------------------------
    def _apply_view_transform(self, pts, center=None):
        # Build rotation matrix
        R_yaw = R.from_euler("y", self.yaw).as_matrix()
        R_pitch = R.from_euler("x", self.pitch).as_matrix()
        R_roll = R.from_euler("z", self.roll).as_matrix()
        Rview = R_roll @ R_pitch @ R_yaw

        if center is not None:
            pts = pts - center

        return pts @ Rview.T

    def _project(self, p):
        x = int(p[0] * self.scale + self.width / 2)
        y = int(p[1] * self.scale + self.height / 2)
        return x, y

    # --------------------------------------------------------------
    #  RENDERING
    # --------------------------------------------------------------
    def render(self, center=None, show_axes=True):
        # Create blank canvas
        img = np.ones((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.bg_color

        # Draw axes
        if show_axes:
            self._draw_axes(img, center)

        # Draw points
        for p, size, color in self.points:
            p3d = self._apply_view_transform(p.reshape(1,3), center)[0]
            x, y = self._project(p3d)
            cv2.circle(img, (x, y), size, color, -1)

        # Draw vectors
        for start, end, color, thick in self.vectors:
            s3d = self._apply_view_transform(start.reshape(1,3), center)[0]
            e3d = self._apply_view_transform(end.reshape(1,3), center)[0]
            sx, sy = self._project(s3d)
            ex, ey = self._project(e3d)
            cv2.arrowedLine(img, (sx, sy), (ex, ey), color, thick, tipLength=0.2)

        # Draw text
        for text, position, color, thickness, scale in self.text:
            x, y = self._project(np.array(position))
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

        # Display
        cv2.imshow("3D Debug View", img)
        self.handle_keys()

    # --------------------------------------------------------------
    #  AXES
    # --------------------------------------------------------------
    def _draw_axes(self, img, center):
        axis_len = 50
        origin = np.array([0,0,0], dtype=float)

        axes = {
            "x": (np.array([axis_len,0,0]), (0,255,0)),
            "y": (np.array([0,axis_len,0]), (255,0,0)),
            "z": (np.array([0,0,axis_len]), (0,128,255))
        }

        for name, (vec, color) in axes.items():
            v_rot = self._apply_view_transform(vec.reshape(1,3), center)[0]
            o_rot = self._apply_view_transform(origin.reshape(1,3), center)[0]

            x1, y1 = self._project(o_rot)
            x2, y2 = self._project(v_rot)

            cv2.line(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(img, name.upper(), (x2+5, y2+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --------------------------------------------------------------
    #  KEYBOARD CONTROLS
    # --------------------------------------------------------------
    def handle_keys(self):
        key = cv2.waitKey(1)
        if key == -1:
            return

        if key == ord('q'): self.yaw -= 0.05
        if key == ord('d'): self.yaw += 0.05
        if key == ord('z'): self.pitch -= 0.05
        if key == ord('s'): self.pitch += 0.05
        if key == ord('a'): self.roll -= 0.05
        if key == ord('e'): self.roll += 0.05
        if key == ord('+'): self.scale *= 1.1
        if key == ord('-'): self.scale /= 1.1
