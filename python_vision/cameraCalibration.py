import cv2
import numpy as np

# -------- PARAMÈTRES --------
CHECKERBOARD = (8, 5)   # 8 coins horizontaux, 5 coins verticaux
SQUARE_SIZE = 20      # taille des cases en mètres (20 mm)
MAX_IMAGES = 16

# Préparation des points 3D du damier (en mètres)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # conversion en mm

objpoints = []
imgpoints = []

# -------- CAPTURE CAM --------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Appuyez sur ESPACE pour capturer une image.")
print("Appuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # ----- Affichage visuel -----
    if found:
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, found)
        status_text = f"Damier détecté ✔️ | Coins: {len(corners)} | Images: {len(objpoints)}/{MAX_IMAGES}"
        color = (0, 255, 0)
    else:
        status_text = f"Damier non détecté ❌ | Images: {len(objpoints)}/{MAX_IMAGES}"
        color = (0, 0, 255)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (40, 40, 40), -1)
    cv2.putText(frame, status_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Calibration", frame)

    key = cv2.waitKey(1) & 0xFF

    # ----- Capture image -----
    if key == 32:  # ESPACE
        if found:
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"[OK] Image capturée ({len(objpoints)}/{MAX_IMAGES})")
        else:
            print("[ERREUR] Damier non détecté, impossible de capturer.")

    # ----- Quitter -----
    if key == ord('q'):
        break

    # ----- Lancer calibration après 16 images -----
    if len(objpoints) == MAX_IMAGES:
        print("\n=== Calibration en cours... ===")
        break

cap.release()
cv2.destroyAllWindows()

# ----- Si pas assez d'images -----
if len(objpoints) < MAX_IMAGES:
    print(f"Seulement {len(objpoints)} images capturées. Calibration annulée.")
    exit()

# -------- CALIBRATION --------
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n=== Résultats ===")
print("Matrice intrinsèque :\n", camera_matrix)
print("Coefficients de distorsion :\n", dist_coeffs)

np.savez("calibration_data.npz",
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs)

print("\n[OK] Calibration enregistrée dans calibration_data.npz")

# -------- TEST UNDISTORT --------
cap = cv2.VideoCapture(0)
print("\nPress 'q' to exit the undistorted preview.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    cv2.imshow("Undistorted", undistorted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
