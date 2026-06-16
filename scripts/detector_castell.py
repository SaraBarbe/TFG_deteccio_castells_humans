import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from pathlib import Path
from funcions import IoU, crop_tronc, choose_classifier, apply_classifier, IoMin
from ultralytics import YOLO
import cv2

# Directoris
img_dir = Path(r"C:\Users\34639\OneDrive\Escriptori\TFG\IMATGES\01.TESTER\TEST_FINAL_FINAL")  
labels_dir_OD = Path(r"C:/Users/34639/OneDrive/Escriptori/TFG/RESULTS/HIBRID/OD_labels_FINAL2")
labels_dir_C = Path(r"C:/Users/34639/OneDrive/Escriptori/TFG/RESULTS/HIBRID/C_labels_FINAL2")
csv_file = Path(r"C:/Users/34639/OneDrive/Escriptori/TFG/RESULTS/HIBRID/TEST/results_test_FINAL3.csv")

# Creem els directoris de sortida si no existeixen
labels_dir_OD.mkdir(parents=True, exist_ok=True)
labels_dir_C.mkdir(parents=True, exist_ok=True)

# Models dels classificadors
model_OD = YOLO(r"C:\Users\34639\OneDrive\Escriptori\TFG\MODELS\entrenats\OD_best_small.pt")
model_C2 = YOLO(r"C:\Users\34639\OneDrive\Escriptori\TFG\MODELS\entrenats\C2_best.pt")
model_C3 = YOLO(r"C:\Users\34639\OneDrive\Escriptori\TFG\MODELS\entrenats\C3_best.pt")
model_C4 = YOLO(r"C:\Users\34639\OneDrive\Escriptori\TFG\MODELS\entrenats\C4_best.pt")

dataset = []
extensions = {".jpg", ".jpeg", ".png"}

# Obtenim totes les imatges de la carpeta
img_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in extensions]

if not img_paths:
    print("No s'han trobat imatges a la carpeta!")
else:
    print(f"S'han trobat {len(img_paths)} imatges. Processant...")

for i, img_path in enumerate(img_paths, 1):
    print(f"[{i}/{len(img_paths)}] Processant: {img_path.name}")
    
    try:
        base_name = img_path.stem

        # Llegim la imatge
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  ⚠ No s'ha pogut llegir la imatge, saltant...")
            continue

        # Apliquem el detector
        OD_results = model_OD(img)[0]

        # Guardem la imatge amb les boxes
        annotated_img = OD_results.plot()
        save_path = labels_dir_OD / f"{base_name}_annotated.jpg"
        cv2.imwrite(str(save_path), annotated_img)

        # Generem i guardem el txt
        txt_path = labels_dir_OD / f"{base_name}.txt"
        with open(txt_path, "w") as f:
            for box in OD_results.boxes:
                class_id = int(box.cls)
                conf     = float(box.conf)
                x, y, w, h = box.xywhn[0].tolist()
                f.write(f"{class_id} {x} {y} {w} {h} {conf}\n")

        # Llegim el txt
        detections = open(txt_path).read().strip().split("\n")

        # Inicialitzem variables per cada imatge
        num_castellers = 0
        num_tronc = 0
        num_pom = 0
        num_pinya = 0
        x_casteller = []
        y_casteller = []
        pinya_box = []
        pom_box = []
        tronc_box = []
        castellers_boxes = []

        for detection in detections:
            if not detection:
                continue

            class_id, x, y, w, h, conf = map(float, detection.split())

            if conf < 0.4:
                continue

            if class_id == 1:
                num_pinya += 1
                pinya_box.append((x, y, w, h))
            elif class_id == 2:
                num_pom += 1
                pom_box.append((x, y, w, h))
            elif class_id == 3:
                num_tronc += 1
                tronc_box.append((x, y, w, h))
            elif class_id == 0:
                castellers_boxes.append((x, y, w, h))

        # Comprovem si el casteller està dins del tronc
        for x, y, w, bh in castellers_boxes:
            inside = False
            for bx, by, bw, bh in tronc_box:
                if (x > bx - bw/2) and (x < bx + bw/2) and (y > by - bh/2) and (y < by + bh/2):
                    inside = True
                    break
            if inside:
                num_castellers += 1
                x_casteller.append(x)
                y_casteller.append(y)


        pinya_box_filtrada = []
        for box in pinya_box:
            px, py, pw, ph = box
            
            if tronc_box:
                tx, ty, tw, th = tronc_box[0] #centre del tronc
            else: #en cas que no es detecti tronc
                tx = 0.5
                tw = 0.5
            
            # La pinya ha d'estar dins del rang horitzontal del tronc + marge
            marge = 0.2 
            if abs(px - tx) < (tw / 2 + marge):
                pinya_box_filtrada.append(box)

        pinya_box = pinya_box_filtrada


        pinya_detected = []
        for box in pinya_box:
            is_duplicated = False
            for valid_box in pinya_detected:
                if IoU(box, valid_box) > 0.5 or IoMin(box, valid_box) > 0.8:
                    is_duplicated = True
                    break
            if not is_duplicated:
                pinya_detected.append(box)

        num_pinya = max(len(pinya_detected), 1)
        num_tronc = max(num_tronc, 1)
        num_pom = max(num_pom, 1)

        # Calculem els pisos del tronc
        pisos_tronc = 0
        castellers_per_pis = []

        if y_casteller:
            clustering = DBSCAN(eps=0.05, min_samples=1).fit(np.array(y_casteller).reshape(-1, 1))
            pisos_tronc = len(set(clustering.labels_))
            for label in set(clustering.labels_):
                count = sum(clustering.labels_ == label)
                castellers_per_pis.append(count)

        alçada_castell = 4 + pisos_tronc
        folre = (num_pinya == 2)
        manilles = (num_pinya == 3)

        if folre:
            alçada_castell += 1
        if manilles:
            alçada_castell += 2

        amplada_clustering = max(castellers_per_pis) if castellers_per_pis else 0

        # Classificador
        amplada_classificador = 0
        conf_cls = 0
        model_cls = choose_classifier(pisos_tronc, model_C2, model_C3, model_C4)
        img_tronc = crop_tronc(img, tronc_box)
        amplada_classificador, conf_cls = apply_classifier(img_tronc, model_cls)

        if amplada_classificador is not None:
            txt_cls_path = labels_dir_C / f"{base_name}_cls.txt"
            with open(txt_cls_path, "w") as f:
                f.write(f"{conf_cls:.2f} {amplada_classificador}\n")

        amplada_castell = max(amplada_classificador, amplada_clustering) if amplada_classificador is not None else amplada_clustering

        # Amplades vàlides 
        amplades_valides = [2, 3, 4, 5, 7]

        # Si l'amplada detectada no és vàlida, utilitzem el classificador
        if amplada_castell not in amplades_valides and amplada_classificador is not None:
            amplada_castell = amplada_classificador

        if amplada_castell > 0:
            castell = f"{amplada_castell}d{alçada_castell}"
            if folre:
                castell += "f"
            elif manilles:
                castell += "fm"
        else:
            castell = "desconegut"

        castell_real = img_path.stem.split("_")[0]

        dataset.append({
            "image_filename": img_path.name,
            "castell_real": castell_real,
            "castell_pred": castell,
            "amplada_clustering": amplada_clustering,
            "amplada_classificador": amplada_classificador,
            "num_castellers": num_castellers,
            "num_tronc": num_tronc,
            "num_pom": num_pom,
            "num_pinya": num_pinya,
            "pisos_tronc": pisos_tronc,
            "castellers_per_pis": castellers_per_pis,
        })

        print(f"  ✓ Castell detectat: {castell}")

    except Exception as e:
        print(f"  ✗ Error processant {img_path.name}: {e}")
        continue

# Guardem el CSV final
if dataset:
    df = pd.DataFrame(dataset)
    df.to_csv(csv_file, mode='a', header=not csv_file.exists(), index=False)
    print(f"\nFet! {len(dataset)} imatges processades → {csv_file}")
else:
    print("\nNo s'ha pogut processar cap imatge.")