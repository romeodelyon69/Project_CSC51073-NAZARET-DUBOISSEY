import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374,
                            380, 381, 382, 362]  # Left eye landmarks
RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145,
                            144, 163, 7]  # Right eye landmarks

#http://benraynal.com/publis/AFIG07nat.pdf
class pupilDetector2007:
    def __init__(self):
        self.iris_color = None
        self.lowThreshMorpho = 0.2
        self.highThreshMorpho = 0.35
        self.threshColor = 0.85
        self.Cc = 0
        self.Cm = 0
        self.bestCm = 0
        self.best_circle = None
        self.current_radius = None

    def detect_iris_color(self, eye_image, circle):
        """Compute the average color inside a reliable circle (strong morpho score)."""
        x, y, r = circle
        mask = np.zeros(eye_image.shape[:2], np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        mean_color = cv2.mean(eye_image, mask=mask)[:3]
        self.iris_color = np.array(mean_color) / 255.0
        return self.iris_color
    
    def compute_Cm(self, edge_image, circle):
        """Morphological coefficient: correlation between contour image and circular mask."""
        x, y, r = circle
        mask = self._circular_mask(r, thickness=1)
        size = mask.shape[0]
        half = size // 2

        # extract region around (x, y)
        x1, y1 = int(x - half), int(y - half)
        x2, y2 = int(x + half + 1), int(y + half + 1)
        if x1 < 0 or y1 < 0 or x2 > edge_image.shape[1] or y2 > edge_image.shape[0]:
            return 0.0  # skip borders

        sub = edge_image[y1:y2, x1:x2].astype(np.float32) / 255.0
        Cm = np.sum(sub * mask) / np.sum(mask)
        return Cm

    def compute_Cc(self, eye_image, circle):
        """Colorimetric coefficient: similarity between candidate and iris color."""
        if self.iris_color is None:
            return 0.0
        x, y, r = circle
        mask = np.zeros(eye_image.shape[:2], np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        mean_color = np.array(cv2.mean(eye_image, mask=mask)[:3]) / 255.0
        dist = np.linalg.norm(mean_color - self.iris_color)
        Cc = 1 - dist / np.sqrt(3)
        return Cc
    
    def pass1(self, eye_image):
        """First pass: rough detection using Canny + circular masks."""
        
        gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        edges_big = cv2.resize(edges, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("edges big", edges_big)

        h, w = gray.shape
        best_score = 0
        best_circle = None

        # scan a range of radii relative to image size
        
        min_r = int(min(h, w) * 0.2)
        max_r = int(min(h, w) * 0.5)
        step_r = max(1, (max_r - min_r) // 10)

        for r in range(min_r, max_r, step_r):
            for y in range(r, h - r, r // 2):
                for x in range(r, w - r, r // 2):
                    Cm = self.compute_Cm(edges, (x, y, r))
                    if Cm > self.bestCm:
                        self.bestCm = Cm
                    if Cm > self.highThreshMorpho and self.iris_color is None:
                        # if strong morphological score -> compute iris color
                        self.detect_iris_color(eye_image, (x, y, r))

                    if Cm > self.lowThreshMorpho:
                        Cc = self.compute_Cc(eye_image, (x, y, r))
                        score = Cm * Cc
                        if score > best_score:
                            best_score = score
                            best_circle = (x, y, r)
                            self.Cm = Cm
                            self.Cc = Cc
                            self.current_radius = r

        
        if best_circle is not None and best_score > 0:
            self.best_circle = best_circle
            return True
        return False

    def pass2(self, eye_image):
        """Second pass: refinement using elliptical masks or gradient radial."""
        if self.best_circle is None:
            return False
        x, y, r = self.best_circle

        # define small window around first result
        margin = int(r * 0.5)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(eye_image.shape[1], x + margin), min(eye_image.shape[0], y + margin)

        sub_img = eye_image[y1:y2, x1:x2]
        gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)

        # gradient magnitude centered around best_circle
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)

        # recompute Cm in a small radius interval
        best_refined = self.best_circle
        best_score = self.Cm * self.Cc
        for dr in [-2, -1, 0, 1, 2]:
            new_r = max(5, r + dr)
            Cm2 = self.compute_Cm((grad > np.mean(grad)).astype(np.uint8) * 255,
                                   (x, y, new_r))
            Cc2 = self.compute_Cc(eye_image, (x, y, new_r))
            score = Cm2 * Cc2
            if score > best_score:
                best_score = score
                best_refined = (x, y, new_r)

        self.best_circle = best_refined
        return True

    def detect_pupil(self, eye_image):
        """Main method to detect pupil in the eye image."""
        "reset state"
        self.bestCm = 0
        self.best_circle = None

        if self.pass1(eye_image):
            if self.pass2(eye_image):
                return self.best_circle
        return None

    def _circular_mask(self, r, thickness=2):
        """Generate Gaussian-weighted circular mask."""
        size = 2 * r + 1
        mask = np.zeros((size, size), dtype=np.float32)
        cv2.circle(mask, (r, r), r, 1, thickness)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        return mask


def fit_ellipse_to_eye(landmarks, eye_landmarks, h , w):
    points = ([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_landmarks])
    if len(points) >= 5:  # fitEllipse requires at least 5 points
        ellipse = cv2.fitEllipse(np.array(points))
        return ellipse, points
    return None, points

def extract_eye_region(frame, ellipse):
    eye_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.ellipse(eye_mask, ellipse, (255, 255, 255), -1)
    eye_region = cv2.bitwise_and(frame, frame, mask=eye_mask)
    return eye_region, eye_mask

def extract_eye_large_region(frame, ellipse, scale=1.2):
    #a square region around the eye; centered on the ellipse center with size the big axe scaled by 'scale'
    eye_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    (x, y), (MA, ma), angle = ellipse
    big_axe = max(MA, ma)
    side_length = int(big_axe * scale)
    top_left = (int(x - side_length // 2), int(y - side_length // 2))
    bottom_right = (int(x + side_length // 2), int(y + side_length // 2))
    cv2.rectangle(eye_mask, top_left, bottom_right, (255, 255, 255), -1)
    eye_region = cv2.bitwise_and(frame, frame, mask=eye_mask)
    print(f"Extracted eye region from {top_left} to {bottom_right}")
    return eye_region, eye_mask

def stretch_contrast_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv_stretched = cv2.merge([h, s, v])
    eye_image_stretched = cv2.cvtColor(hsv_stretched, cv2.COLOR_HSV2BGR)
    return eye_image_stretched


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)
left_pupil_detector = pupilDetector2007()
right_pupil_detector = pupilDetector2007()

while cap.isOpened():
    success, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    h, w, _ = frame.shape

    print("left pupil detector iris color:", left_pupil_detector.iris_color)
    print("right pupil detector iris color:", right_pupil_detector.iris_color)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            #fit ellipse to left eye points
            left_eye_ellipse, left_eye_points = fit_ellipse_to_eye(face_landmarks.landmark, LEFT_EYE_LANDMARKS, h, w)
            right_eye_ellipse, right_eye_points = fit_ellipse_to_eye(face_landmarks.landmark, RIGHT_EYE_LANDMARKS, h, w)
            # extraie deux images au niveau des ellipses

            if left_eye_ellipse is not None and right_eye_ellipse is not None:
                # Crée un masque pour les yeux
                left_eye_img, left_eye_mask = extract_eye_large_region(frame, left_eye_ellipse)
                right_eye_img, right_eye_mask = extract_eye_large_region(frame, right_eye_ellipse)
                # Combine les deux images des yeux
                eyes = cv2.bitwise_or(left_eye_img, right_eye_img)
                
                # Recadre autour de l'oeil gauche /droite, en se basant sur left_eye_mask et right_eye_mask

                x, y, w_box, h_box = cv2.boundingRect(left_eye_mask)
                left_eye_img = left_eye_img[y:y+h_box, x:x+w_box]
                x, y, w_box, h_box = cv2.boundingRect(right_eye_mask)
                right_eye_img = right_eye_img[y:y+h_box, x:x+w_box]

                #cv2.imshow(cv2.cvtColor(left_eye_img, cv2.COLOR_BGR2RGB))
                #cv2.imshow(cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2RGB))

                # #étire l'histogramme pour améliorer le contraste en utilisant CLAHE
                # # Convertir en HSV
                # constrated_left_eye = stretch_contrast_hsv(left_eye_img.copy())
                # constrated_right_eye = stretch_contrast_hsv(right_eye_img.copy())
                # # Affiche les résultats
                # imshow(cv2.cvtColor(constrated_left_eye, cv2.COLOR_BGR2RGB))
                # imshow(cv2.cvtColor(constrated_right_eye, cv2.COLOR_BGR2RGB))

                lb = left_pupil_detector.detect_pupil(left_eye_img)
                if left_pupil_detector.best_circle is not None:
                    l_color = (255 * left_pupil_detector.iris_color)
                    x, y, r = map(int, left_pupil_detector.best_circle)
                    cv2.circle(left_eye_img, (x, y), r, l_color, 2)
                    left_eye_big = cv2.resize(left_eye_img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("left eye", left_eye_big)
                                
                lr = right_pupil_detector.detect_pupil(right_eye_img)
                if right_pupil_detector.best_circle is not None:
                    r_color = (255 * right_pupil_detector.iris_color)
                    x, y, r = map(int, right_pupil_detector.best_circle)
                    cv2.circle(right_eye_img, (x, y), r, r_color, 2)
                    right_eye_img_big = cv2.resize(right_eye_img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("right eye", right_eye_img_big)

                #write the left Cc value on the frame
                cv2.putText(frame, f"Left Cc: {left_pupil_detector.Cc:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Cc: {right_pupil_detector.Cc:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #draw the circle on the frame, adjusted to the eye position in the frame
                if left_pupil_detector.best_circle is not None:
                    x, y, r = map(int, left_pupil_detector.best_circle)
                    ex, ey, ew, eh = cv2.boundingRect(left_eye_mask)
                    cv2.circle(frame, (x + ex, y + ey), r, (0, 255, 0), 2)
                if right_pupil_detector.best_circle is not None:
                    x, y, r = map(int, right_pupil_detector.best_circle)
                    ex, ey, ew, eh = cv2.boundingRect(right_eye_mask)
                    cv2.circle(frame, (x + ex, y + ey), r, (0, 255, 0), 2)
                cv2.imshow("face", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Échap pour quitter
        break
cap.release()
cv2.destroyAllWindows()