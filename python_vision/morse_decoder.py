from datetime import datetime, timedelta
MIN_SAME_STATE_COUNT = 2
DELTA_TIME_NOT_BLANK = 1

# 1 for closed 0 for open
# for signs, 0 is a dot, 1 is a dash

# dict from signs to letters or spaces
MORSE_DICT = {
    (0,): 'E',           # .
    (1,): 'T',           # -
    (0, 0): 'I',         # ..
    (0, 1): 'A',         # .-
    (1, 0): 'N',         # -.
    (1, 1): 'M',         # --
    (0, 0, 0): 'S',      # ...
    (0, 0, 1): 'U',      # ..-
    (0, 1, 0): 'R',      # .-.
    (0, 1, 1): 'W',      # .--
    (1, 0, 0): 'D',      # -..
    (1, 0, 1): 'K',      # -.-
    (1, 1, 0): 'G',      # --.
    (1, 1, 1): 'O',      # ---
    (0, 0, 0, 0): 'H',   # ....
    (0, 0, 0, 1): 'V',   # ...-
    (0, 0, 1, 0): 'F',   # ..-.
    (0, 1, 0, 0): 'L',   # .-..
    (0, 1, 1, 0): 'P',   # .--.
    (0, 1, 1, 1): 'J',   # .---
    (1, 0, 0, 0): 'B',   # -...
    (1, 0, 0, 1): 'X',   # -..-
    (1, 0, 1, 0): 'C',   # -.-.
    (1, 0, 1, 1): 'Y',   # -.--
    (1, 1, 0, 0): 'Z',   # --..
    (1, 1, 0, 1): 'Q',   # --.-
}

class MorseDecoder:

    sign_stack:tuple[bool,...]

    last_left :bool
    last_right : bool
    same_state_count : int
    last_not_blank : datetime

    def __init__(self):
        self.sign_stack = ()
        self.last_left = False
        self.last_right = False
        self.same_state_count = 0
        self.last_not_blank = datetime.now()
    
    def add_entry(self, left:bool, right:bool):
        if left == self.last_left and right == self.last_right:
            self.same_state_count += 1
        else:
            self.same_state_count = 0
            self.last_left = left
            self.last_right = right
        if left or right:
            self.last_not_blank = datetime.now()
    
        if self.same_state_count == MIN_SAME_STATE_COUNT:
            if left and right:
                sign = True
            elif left and not right:
                sign = False
            else:
                return
            new_signs = (*self.sign_stack, sign)
            self.sign_stack = new_signs

        if datetime.now() - self.last_not_blank > timedelta(seconds=DELTA_TIME_NOT_BLANK):
            if self.sign_stack in MORSE_DICT:
                letter = MORSE_DICT[self.sign_stack]
                self.sign_stack = ()
                return letter
            else:
                self.sign_stack = ()
        return None