import cv2
from ultralytics import YOLO
import os


# ---------- Cấu hình ----------
MODEL_PATH = r"E:/MY PROJECTS/AI/Pedestrian Objection/runs/detect/train/weights/best.pt"
INPUT_FOLDER = r"E:/MY PROJECTS/AI/input_images"
OUTPUT_FOLDER = r"E:/MY PROJECTS/AI/output_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Load model
model = YOLO(MODEL_PATH)


# Hàm tính IoU giữa 2 bbox
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)


# ---------- Xử lý folder ảnh ----------
for img_name in os.listdir(INPUT_FOLDER):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue


    img_path = os.path.join(INPUT_FOLDER, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print("Không đọc được ảnh:", img_name)
        continue


    results = model(img)
    boxes = results[0].boxes
    annotated_img = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)


    ped_wrong_boxes = []
    ped_correct_boxes = []
    light_red_boxes = []
    light_green_boxes = []


    # Phân loại bounding box và in thông tin
    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])


        if label == "ped_wrong":
            ped_wrong_boxes.append((x1, y1, x2, y2))
        elif label == "ped_correct":
            ped_correct_boxes.append((x1, y1, x2, y2))
        elif label == "light_red":
            light_red_boxes.append((x1, y1, x2, y2))
        elif label == "light_green":
            light_green_boxes.append((x1, y1, x2, y2))


        print(f"Image {img_name}: Class: {cls_id} ({label}), Confidence: {conf:.2f}, Box: {(x1, y1, x2, y2)}")


    # Hàm kiểm tra vi phạm theo logic mới
    def check_violation(ped_box, ped_type):
        # ped_wrong → luôn tính vi phạm
        if ped_type == "ped_wrong":
            return True
        # ped_correct → tính vi phạm chỉ khi có đèn xanh gần bbox
        elif ped_type == "ped_correct":
            for green_box in light_green_boxes:
                if iou(ped_box, green_box) > 0.1:  # ngưỡng IoU
                    return True
            return False
        return False


    # Đếm vi phạm
    violation_count = 0
    for ped in ped_wrong_boxes:
        if check_violation(ped, "ped_wrong"):
            violation_count += 1
    for ped in ped_correct_boxes:
        if check_violation(ped, "ped_correct"):
            violation_count += 1


    # Vẽ số lượng vi phạm lên ảnh
    if violation_count > 0:
        cv2.putText(annotated_img, f"VIOLATIONS: {violation_count}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


    # Lưu ảnh đã annotate
    output_path = os.path.join(OUTPUT_FOLDER, img_name)
    cv2.imwrite(output_path, annotated_img)


print(f"Processing done! Images saved as: {OUTPUT_FOLDER}")
