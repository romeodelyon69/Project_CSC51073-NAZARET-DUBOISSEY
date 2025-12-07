import mouse 
import pyautogui


class MouseController:
    def __init__(self):
        self.u = 0.0
        self.v = 0.0

        self.screen_width, self.screen_height = pyautogui.size()

    def update_mouse_position(self, u: float, v: float) -> None:
        self.u = u
        self.v = v

    def move_mouse(self) -> None:
        mouse.move(self.u * self.screen_width, self.v * self.screen_height)

    def update_and_move(self, u: float, v: float) -> None:
        self.update_mouse_position(u, v)
        self.move_mouse()