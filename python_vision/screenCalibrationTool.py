import cv2 
import numpy as np
import pyautogui

class CalibrationTool:
    def __init__(self, nb_points_row=4, nb_points_col=4):
        self.screen_points_3D = []      # List of 3D points for screen corners
        self.normalized_screen_points_2D = []  # Corresponding normalized 2D points
        self.is_calibrated = False
        self.is_calibrating = False
        self.nb_point_row = nb_points_row
        self.nb_point_col = nb_points_col
        self.current_calib_index = 0
        self.left_indicators = []
        self.right_indicators = []

        self.calib_image_size = (pyautogui.size().width, pyautogui.size().height)
        self.calib_window_name = "Calibration"
        self.calib_image = np.zeros((self.calib_image_size[1], self.calib_image_size[0], 3), dtype=np.uint8)

    def add_screen_point(self, point_3D = None, left_indicator=None, right_indicator=None):
        normalized_point_2D = (
            (self.current_calib_index % self.nb_point_col + 0.5) / self.nb_point_col,
            (self.current_calib_index // self.nb_point_col + 0.5) / self.nb_point_row,
        )
        self.screen_points_3D.append(point_3D)
        self.normalized_screen_points_2D.append(normalized_point_2D)
        self.left_indicators.append(left_indicator)
        self.right_indicators.append(right_indicator)

    def start_calibration(self):
        self.is_calibrated = False
        self.is_calibrating = True
        self.current_calib_index = 0
        self.calib_image[:] = 0
        cv2.namedWindow(self.calib_window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.calib_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    def show_image(self):
        if self.current_calib_index < self.nb_point_row * self.nb_point_col:
            # Clear image
            self.calib_image[:] = 0

            # Compute position of the calibration point
            row = self.current_calib_index // self.nb_point_col
            col = self.current_calib_index % self.nb_point_col
            x = int((col + 0.5) * self.calib_image_size[0] / self.nb_point_col)
            y = int((row + 0.5) * self.calib_image_size[1] / self.nb_point_row)

            # Draw calibration point
            cv2.circle(self.calib_image, (x, y), 20, (0, 255, 0), -1)

            # Show image
            cv2.imshow(self.calib_window_name, self.calib_image)
            
    def next_calibration_point(self):
        self.current_calib_index += 1
        if self.current_calib_index >= self.nb_point_row * self.nb_point_col:
            self.is_calibrated = True
            self.is_calibrating = False
            cv2.destroyWindow(self.calib_window_name)
            return False
        return True

    def get_calibration_data(self):
        return {"screen_3D_calib_points": self.screen_points_3D, "screen_uv_calib_points": self.normalized_screen_points_2D, "left_indicators": self.left_indicators, "right_indicators": self.right_indicators}