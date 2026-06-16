def IoU(box1, box2):

    x1, y1, w1, h1 = box1  #(x_center, y_center, width, height)
    x2, y2, w2, h2 = box2

    #Calculem les cantonades
    xa1, ya1 = x1 - w1/2, y1 - h1/2
    xa2, ya2 = x1 + w1/2, y1 + h1/2

    xb1, yb1 = x2 - w2/2, y2 - h2/2
    xb2, yb2 = x2 + w2/2, y2 + h2/2

    #Calculem les interseccions
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    #Àrea de la intersecció
    inter_area = inter_w * inter_h

    #Calculem les àrees individuals
    area1 = w1 * h1
    area2 = w2 * h2

    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0

def yolo_to_pixels(box, img_w, img_h):
    x, y, w, h = box
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return x1, y1, x2, y2

def crop_tronc(img, tronc_box):
    
    if len(tronc_box) == 0:
        return None

    img_h, img_w = img.shape[:2]

    # En cas de tenir 2 troncs detectats, triem la bbox amb més àrea
    millor_box = max(tronc_box, key=lambda b: b[2] * b[3])

    x1, y1, x2, y2 = yolo_to_pixels(millor_box, img_w, img_h)
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop

def choose_classifier(pisos_tronc, model_C2, model_C3, model_C4):
        if pisos_tronc == 2:
            return model_C2
        elif pisos_tronc == 3:
            return model_C3
        elif pisos_tronc == 4:
            return model_C4
        else:
            return None
        
def apply_classifier(crop_img, model):
    if crop_img is None or model is None:
        return None, None

    results = model.predict(source=crop_img, verbose=False)
    pred_idx = int(results[0].probs.top1)
    conf = float(results[0].probs.top1conf)

    classe_real = int(model.names[pred_idx])
    return classe_real, conf

def IoMin(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa1, ya1 = x1 - w1/2, y1 - h1/2
    xa2, ya2 = x1 + w1/2, y1 + h1/2

    xb1, yb1 = x2 - w2/2, y2 - h2/2
    xb2, yb2 = x2 + w2/2, y2 + h2/2

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h
    min_area = min(w1 * h1, w2 * h2)

    return inter_area / min_area if min_area > 0 else 0
    

    