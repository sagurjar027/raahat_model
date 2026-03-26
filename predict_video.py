from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, Counter

model_path = "best2.pt"
model = YOLO(model_path)
def raahat_predict_video(input_video_path, output_video_path, line):
    cap = cv2.VideoCapture(input_video_path)
#  -------------  model paths and load model----------------------
    model_path = "best2.pt"
    model = YOLO(model_path)
# ----------------model loaded----------------------------- + len()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"🚀 Processing video for Line {line}...")

    prev_positions = {}
    speeds = []
    track_class_history = defaultdict(list)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        boxes = results[0].boxes

        if boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for i, track_id in enumerate(ids):

                class_id = int(classes[i])
                confidence = confs[i]

                # ✅ Confidence filter
                if confidence >= 0.6:
                    track_class_history[track_id].append(class_id)

                # 🔹 Speed
                x1, y1, x2, y2 = xyxy[i]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if track_id in prev_positions:
                    prev_cx, prev_cy = prev_positions[track_id]
                    dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                    speeds.append(dist * fps)

                prev_positions[track_id] = (cx, cy)

        out.write(annotated_frame)

    cap.release()
    out.release()

    # 🔥 Final classification
    final_class_per_id = {}

    for track_id, class_list in track_class_history.items():
        if not class_list:
            continue
        final_class = Counter(class_list).most_common(1)[0][0]
        final_class_per_id[track_id] = final_class

    # 🔥 Split
    EMERGENCY_CLASSES = [0, 2]

    unique_ids = set()
    emergency_ids = set()

    for track_id, final_class in final_class_per_id.items():
        if final_class in EMERGENCY_CLASSES:
            emergency_ids.add(track_id)
        else:
            unique_ids.add(track_id)
# total_vehicle_count = len(unique_ids) + len(emergency_ids)
    total_vehicle_count = len(unique_ids)
    emergency_count = len(emergency_ids)

    # 🔹 Density
    if total_vehicle_count < 5:
        density_label = "low"
    elif total_vehicle_count < 15:
        density_label = "medium"
    else:
        density_label = "high"

    # 🔹 Speed
    PIXEL_TO_METER = 0.05
    avg_speed_pixel_per_sec = float(np.mean(speeds)) if speeds else 0.0
    avg_speed = avg_speed_pixel_per_sec * PIXEL_TO_METER * 3.6
    
    # 🔹 Emergency
    emergency_flag = "true" if emergency_count >= 1 else "false"

    return {
        "line": line,
        "vehicle_count": total_vehicle_count,
        "density": density_label,
        "avg_speed": round(avg_speed, 2),
        "emergency_video": emergency_flag
    }