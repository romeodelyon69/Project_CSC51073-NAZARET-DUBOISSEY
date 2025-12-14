from morse_decoder import MorseDecoder
import eyeTracker.eyeToolKit as etk


class PupilDecoder:
    morse_decoder = MorseDecoder()

    def __init__(self):
        self.pupil_decoder = None

    def add_entry(
        self,
        left_eye_points: list[tuple[int, int]],
        right_eye_points: list[tuple[int, int]],
    ):
        left_eye_area = etk.get_eye_area(left_eye_points)
        right_eye_area = etk.get_eye_area(right_eye_points)
        left_eye_is_closed = etk.get_eye_is_closed(left_eye_area)
        right_eye_is_closed = etk.get_eye_is_closed(right_eye_area)
        # print(f"left eye is closed: {left_eye_is_closed}({left_eye_area}), right eye is closed: {right_eye_is_closed}({right_eye_area})")

        result = self.morse_decoder.add_entry(left_eye_is_closed, right_eye_is_closed)
        # print(f"sign stack: {morse_decoder.sign_stack}, l: {int(morse_decoder.last_left)}, r: {int(morse_decoder.last_right)}, count: {morse_decoder.same_state_count} and dt {round((datetime.now() - morse_decoder.last_not_blank).total_seconds(), 2)}")
        if result is not None:
            print(f"result: {result}")
