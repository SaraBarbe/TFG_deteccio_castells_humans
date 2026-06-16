import cv2
import os
from pathlib import Path
import numpy as np

#directoris
img_dir = Path(r"C:/Users/34639/OneDrive/Escriptori/TFG/Castells_de_8")
labels_dir = Path(r"C:/Users/34639/OneDrive/Escriptori/TFG/RESULTS/OBJECT_DETECTION/castells_de_8_test/labels")
crops_dir = Path(r"C:/Users/34639/OneDrive/Escriptori/TFG/IMATGES/Troncs_de_4_pisos/noves")

tronc_class_id = 3; #ID de l'object tronc

#Recorrem totes les labels
for file in os.listdir(labels_dir):
    if not file.endswith(".txt"):
        continue

    #associem cada label a la seva imatge
    base_name = file.replace(".txt", "")
    extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

    img_path = None

    for ext in extensions:
        candidate = img_dir / f"{base_name}{ext}"
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        print(f"Imatge no trobada: {file}")
        continue

    #Llegim les deteccions
    detections = open(os.path.join(labels_dir, file)).read().strip().split("\n")

    #Llegim les imatges
    img = cv2.imdecode(
        np.fromfile(str(img_path), dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    img_h, img_w = img.shape[:2]


    for detection in detections:
        detection = detection.strip() #eliminem espais
        if not detection:
            continue

        values = detection.split() #separem la linia en valors
        if len(values) < 5:
            continue

        # Assignem cada valor del .txt
        class_id, x, y, w, h = map(float, values[:5])

        if class_id != tronc_class_id:  #Ignorem les bbox que no son tronc
            continue

        #Coordenades YOLO (normalitzades)
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)

        #Convertim YOLO → píxels
        x1 = int((x - w / 2) * img_w)
        y1 = int((y - h / 2) * img_h)
        x2 = int((x + w / 2) * img_w)
        y2 = int((y + h / 2) * img_h)

        #Assegurem que no surti fora de la imatge
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        #Fem el crop del tronc
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        #Nom del fitxer de sortida
        base_name = img_path.stem
        crop_name = crops_dir / f"{base_name}_tronc.jpg"

        #Guardem el crop
        cv2.imwrite(str(crop_name), crop)

        
