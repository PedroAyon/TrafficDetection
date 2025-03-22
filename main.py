from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Initialize the model and video capture.
model = YOLO("yolo11n.pt")
video_path = "/home/pedro-ayon/dev/ComputoUbicuo/Proyecto/TrafficDetection/video/1.webm"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully.
if not cap.isOpened():
    print("Error: Video file not found or cannot be opened!")
    exit(1)

# Retrieve video FPS (frames per second)
fps = cap.get(cv2.CAP_PROP_FPS)  # e.g., 30 fps => 1/30 seconds between frames

# Base conversion factor (meters per pixel) for a vehicle at a reference distance.
conversion_factor = 0.05  # Example: 0.05 meters per pixel (calibrate for your setup)
# Reference bounding box height from calibration (in pixels).
ref_bbox_height = 40  # Adjust based on a known distance/calibration

# Dictionaries for tracking vehicle histories, sides, and cooldown counts.
track_history = defaultdict(list)
last_side = {}
last_counted_time = {}

vehicle_count = 0
cooldown_duration = 2.0  # Cooldown duration in seconds

# Define the position of the counting line using a sample frame.
ret, sample_frame = cap.read()
if not ret:
    print("Error: Cannot read a frame from the video.")
    exit(1)
frame_height, frame_width = sample_frame.shape[:2]
line_y = int(frame_height * 0.7)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the first frame

# Define maximum display dimensions.
max_display_width = 1600
max_display_height = 1600

# Define vehicle classes (adjust based on your model's class names).
vehicle_classes = {"car", "truck", "bus", "motorcycle", "van"}


def compute_speed(track, fps, conv_factor):
    """
    Compute the vehicle speed in km/h based on the last two positions.
    :param track: List of (x, y) positions.
    :param fps: Frames per second of the video.
    :param conv_factor: Conversion factor (meters per pixel) adjusted for current perspective.
    :return: Speed in km/h.
    """
    if len(track) >= 2:
        (x1, y1) = track[-2]
        (x2, y2) = track[-1]
        pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # Convert pixel distance to meters using the adjusted conversion factor.
        distance_meters = pixel_distance * conv_factor
        speed_mps = distance_meters * fps  # meters per second
        speed_kmph = speed_mps * 3.6  # convert to km/h
        return speed_kmph
    return 0.0


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True)
    annotated_frame = frame.copy()
    cv2.line(annotated_frame, (0, line_y), (frame_width, line_y), (0, 0, 255), 4)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()  # center_x, center_y, width, height
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.cpu().tolist()  # predicted class indices

        for box, track_id, cls in zip(boxes, track_ids, classes):
            label = model.names[int(cls)]
            if label not in vehicle_classes:
                continue

            # Extract center coordinates and dimensions.
            center_x = float(box[0])
            center_y = float(box[1])
            w = float(box[2])
            h = float(box[3])
            x1 = int(center_x - w / 2)
            y1 = int(center_y - h / 2)
            x2 = int(center_x + w / 2)
            y2 = int(center_y + h / 2)

            # Draw bounding box and label.
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (230, 230, 230), 5)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (230, 230, 230), 4)

            # Update track history.
            track = track_history[track_id]
            track.append((center_x, center_y))
            if len(track) > 30:
                track.pop(0)
            if len(track) >= 2:
                points = np.array(track, np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)

            # Dynamically adjust the conversion factor using the bounding box height.
            # If the object is larger (closer), the conversion factor is reduced.
            adjusted_conv_factor = conversion_factor * (ref_bbox_height / h)

            # Compute and display speed.
            speed = compute_speed(track, fps, adjusted_conv_factor)
            cv2.putText(annotated_frame, f"{speed:.1f} km/h", (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # Determine current side (0 if above the line, 1 if below).
            current_side = 0 if center_y < line_y else 1

            # Count vehicles crossing the line with a cooldown.
            if track_id in last_side:
                if current_side != last_side[track_id]:
                    current_time = time.time()
                    if track_id not in last_counted_time or (
                            current_time - last_counted_time[track_id]) > cooldown_duration:
                        vehicle_count += 1
                        last_counted_time[track_id] = current_time
                    last_side[track_id] = current_side
            else:
                last_side[track_id] = current_side

    cv2.putText(annotated_frame, f"Count: {vehicle_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 8)

    disp_h, disp_w = annotated_frame.shape[:2]
    scale = min(max_display_width / disp_w, max_display_height / disp_h, 1.0)
    if scale < 1.0:
        annotated_frame = cv2.resize(annotated_frame, (int(disp_w * scale), int(disp_h * scale)),
                                     interpolation=cv2.INTER_AREA)

    cv2.imshow("YOLO11 Tracking with Speed Adjustment", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
