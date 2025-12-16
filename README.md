# eyeTracker

eyeTracker implémente le pipeline complet d'eye tracking. Tout le code est disponible dans le dossier `eyeTracker`. Pour le lancer, il suffit d'exécuter :

python eyeTracking.py

## Composants principaux

- cameraCalibration.py réalise la calibration de la caméra avec le damier calib.io_checker_200x150_6x9_20 : à réaliser avant d'éxectuer eyetracking.py sous peine de résultat approxiamtif. 
- faceAnalyser : réalise l'essentiel des calculs, en s'appuyant sur screenCalibrationTool pour exécuter la procédure de calibration de l'écran
- mouseController, renderer3D, click_manager : permettent l'interaction avec l'utilisateur et affichent une fenêtre de debug pour visualiser la reconstruction 3D.
- Les autres fichiers, Nlib.py, landmarks.py; eyeToolKit.py sont là comme des mini librairies pour faciliter l'écriture du code

## Contrôles disponibles

1. Regarder la caméra et appuyer sur `r` pour commencer : cela calibre les vecteurs du regard.
2. Appuyer sur `c` pour lancer la calibration : des points apparaissent à l'écran, il faut les regarder et appuyer sur `espace`.
3. Appuyer sur `m` pour activer/désactiver le contrôle de la souris par le regard.
4. Appuyer sur `b` pour changer la méthode utilisée : premièrement reconstruction 3D, puis ICIAP2009, puis la pointe du nez.
5. Déplacement et zoom dans la reconstruction 3D : `z`, `q`, `s`, `d`, `w`, `+`, `-`. Appuyer sur `f` pour alterner le point de référence entre la caméra et le centre du visage.
6. Cliquer : clic gauche = clignement gauche, clic droit = clignement droit.

---

# Morse

Pour exécuter le morse_decoder, lancez :

python firstTest.py, d'autres fenêtre vont s'ouvrir, il suffit des les réduire et vous pouvez commencer à cligner des yeux pour écrire des caractères (dans le terminal)

---

# Autres implémentations

- Pour visualiser les premiers résultats de tracking de pupille faits à la main, exécutez `firstTest.py`.
- Pour explorer plus en détail le pipeline, lancez le notebook `sandbox.ipynb`.
- Pour voir l’implémentation de l’article AFIG 2007 pour le calcul de la position de la pupille, lancez `implementationAlgo2007.py`.
