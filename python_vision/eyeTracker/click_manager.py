import eyeToolKit as etk
import mouse 
import pyautogui
import Nlib

class ClickManager:
    def __init__(self):
        self.pupil_decoder = None
        self.mouse_pos_buffer = Nlib.VecBuffer(9)
        self.last_mouse_position = None
        self.nb_consecutive_frames_left_eye_closed = 0
        self.nb_consecutive_frames_right_eye_closed = 0

    def add_entry(
        self,
        left_eye_points: list[tuple[int, int]],
        right_eye_points: list[tuple[int, int]],
        mouse_position: tuple[float, float]=None,
    ):
        left_eye_area = etk.get_eye_area(left_eye_points)
        right_eye_area = etk.get_eye_area(right_eye_points)
        left_eye_is_closed = etk.get_eye_is_closed(left_eye_area)
        right_eye_is_closed = etk.get_eye_is_closed(right_eye_area)
        # print(f"left eye is closed: {left_eye_is_closed}({left_eye_area}), right eye is closed: {right_eye_is_closed}({right_eye_area})")

        if left_eye_is_closed:
            self.nb_consecutive_frames_left_eye_closed += 1
        else:
            self.nb_consecutive_frames_left_eye_closed = 0
        if right_eye_is_closed:
            self.nb_consecutive_frames_right_eye_closed += 1
        else:
            self.nb_consecutive_frames_right_eye_closed = 0

        #on sauve la dernière position avant le début du clignement (le clignement 
        #peut faire bouger la tête et donc la position du curseur)
        if left_eye_is_closed == 1 or right_eye_is_closed == 1:
            self.last_mouse_position = self.mouse_pos_buffer.get_mean()

        # If left eye is closed for 3 consecutive frames, perform left click
        if self.nb_consecutive_frames_left_eye_closed == 3:
            #move mouse to average position before clicking
            mouse.move(self.last_mouse_position[0], self.last_mouse_position[1])
            mouse.click(button='left')
            print("Left click")
        # If right eye is closed for 3 consecutive frames, perform right click
        if self.nb_consecutive_frames_right_eye_closed == 3:
            #move mouse to average position before clicking
            mouse.move(self.last_mouse_position[0], self.last_mouse_position[1])
            mouse.click(button='right')
            print("Right click")

        self.mouse_pos_buffer.add(mouse_position)

        
