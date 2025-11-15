EyeSize = 0.024  # Approximate diameter of the human eye in meters
class face:
    def __init__(self, landmarks, yaw, pitch, roll):
        self.nose = landmarks["nose"]
        self.left_eye = landmarks["left_eye"]
        self.right_eye = landmarks["right_eye"]
        self.left_pupil = landmarks["left_pupil"]
        self.right_pupil = landmarks["right_pupil"]
        self.orientation = {"yaw": yaw, "pitch": pitch, "roll": roll}
        self.left_eye_vector = None
        self.right_eye_vector = None
        self.left_eyeball= None
        self.right_eyball = None

    def compute_eyeball_positions(self):
        # Estimate eyeball positions based on eye landmarks, eye size, and face orientation
        self.left_eyeball = (
            self.left_eye[0],
            self.left_eye[1],
            self.left_eye[2] - EyeSize / 2,
        )
        self.right_eyeball = (
            self.right_eye[0],
            self.right_eye[1],
            self.right_eye[2] - EyeSize / 2,
        )

    def compute_eye_vectors(self):
        # Compute eye direction vectors based on landmarks
        self.left_eye_vector = (
            self.left_pupil[0] - self.left_eyeball[0],
            self.left_pupil[1] - self.left_eyeball[1],
            self.left_pupil[2] - self.left_eyeball[2],
        )
        self.right_eye_vector = (
            self.right_pupil[0] - self.right_eyeball[0],
            self.right_pupil[1] - self.right_eyeball[1],
            self.right_pupil[2] - self.right_eyeball[2],
        )