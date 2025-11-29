import cv2
from ultralytics import YOLO

# Load model
model = YOLO(r"E:/MY PROJECTS/AI/Pedestrian Objection/runs/detect/train/weights/best.pt")

# Input video path
input_video_path = r"C:/Users/nguye/Downloads/Đèn đỏ_đi sai vạch.mp4"

# MỞ VIDEO 
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Không mở được video!")
    exit()

# Output video path
output_path = "output_violation_đi sai vạch3.video.mp4"

# Lấy thông số video gốc
fps = int(cap.get(cv2.CAP_PROP_FPS)) ## khung hình trên giây
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Khởi tạo VideoWriter để ghi video mới
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Hàm IoU
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame) ## gửi ảnh img lên model để dự đoán
    boxes = results[0].boxes ## lấy các bbox từ kết quả dự đoán
    annotated_frame = results[0].plot() ## vẽ bbox lên ảnh

    ped_wrong_boxes = []
    ped_correct_boxes = []
    light_red_boxes = []
    light_green_boxes = []
    crosswalk_boxes = []

    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "ped_wrong":
            ped_wrong_boxes.append((x1, y1, x2, y2))
        elif label == "ped_correct":
            ped_correct_boxes.append((x1, y1, x2, y2))
        elif label == "light_red":
            light_red_boxes.append((x1, y1, x2, y2))
        elif label == "light_green":
            light_green_boxes.append((x1, y1, x2, y2))
        elif label == "crosswalk":
            crosswalk_boxes.append((x1, y1, x2, y2))

        print(f"Frame {frame_idx}: Class: {cls_id} ({label}), Box: {(x1, y1, x2, y2)}")

    def check_violation(ped_box, ped_type):
        if ped_type == "ped_wrong":
            return True
        elif ped_type == "ped_correct":
            for green_box in light_green_boxes:
                if iou(ped_box, green_box) > 0.1:
                    return True
            return False
        return False

    violation_count = 0
    for ped in ped_wrong_boxes:
        if check_violation(ped, "ped_wrong"):
            violation_count += 1
    for ped in ped_correct_boxes:
        if check_violation(ped, "ped_correct"):
            violation_count += 1

    if violation_count > 0:
        cv2.putText(
            annotated_frame,
            f"VIOLATIONS: {violation_count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3
        )

    out.write(annotated_frame)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing done! Video saved as", output_path)
