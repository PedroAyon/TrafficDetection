import cv2
import numpy as np
import time
from collections import defaultdict

class TrafficDetector:
    def __init__(self, video_path: str, model, conversion_factor: float = 0.05,
                 ref_bbox_height: int = 40, line_orientation: str = "horizontal",
                 line_position_ratio: float = 0.7, show_video: bool = False,
                 cooldown_duration: float = 2.0, vehicle_classes: set = None,
                 frame_skip: int = 1, target_width: int = 1280, target_height: int = 720):
        """
        Initialize the traffic detector.

        :param video_path: Path to the video file.
        :param model: The AI model instance (e.g., YOLO) used for detection.
        :param conversion_factor: Base conversion factor (meters per pixel).
        :param ref_bbox_height: Reference bounding box height (pixels) for calibration.
        :param line_orientation: 'horizontal' or 'vertical' for the counting line.
        :param line_position_ratio: For horizontal, a fraction of frame height;
                                    for vertical, a fraction of frame width.
        :param show_video: If True, display the annotated video during processing.
        :param cooldown_duration: Minimum time (in seconds) between consecutive counts per vehicle.
        :param vehicle_classes: Set of vehicle class names to detect.
        :param frame_skip: Process every nth frame (e.g., 1=every frame, 2=every other frame).
        :param target_width: The target width for the output video frame.
        :param target_height: The target height for the output video frame.
                             If the original video is smaller, the dimensions are not changed.
        """
        self.video_path = video_path
        self.model = model
        self.conversion_factor = conversion_factor
        self.ref_bbox_height = ref_bbox_height
        self.line_orientation = line_orientation
        self.line_position_ratio = line_position_ratio
        self.show_video = show_video
        self.cooldown_duration = cooldown_duration
        self.vehicle_classes = vehicle_classes or {"car", "truck", "bus", "motorcycle", "van"}
        self.frame_skip = frame_skip
        self.target_width = target_width
        self.target_height = target_height
        self.track_history = defaultdict(list)
        self.last_side = {}
        self.last_counted_time = {}
        self.vehicle_count = 0
        self.speeds = []  # list to store speed values for averaging

    @staticmethod
    def compute_speed(track: list, fps: float, conv_factor: float,
                      ref_bbox_height: float, box_height: float, frame_skip: int = 1) -> float:
        """
        Compute speed in km/h based on the last two positions in the track.
        The speed is corrected by dividing by the frame_skip factor.
        """
        conv_factor = conv_factor * ref_bbox_height / box_height
        if len(track) >= 2:
            (x1, y1), (x2, y2) = track[-2], track[-1]
            pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distance_meters = pixel_distance * conv_factor
            speed_mps = distance_meters * fps
            return (speed_mps * 3.6) / frame_skip  # km/h
        return 0.0

    def process_video(self) -> dict:
        """
        Process the video file and detect/calculate traffic metrics.

        Returns:
            A dictionary with 'vehicle_count' and 'average_speed' (in km/h).
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError("Video file not found or cannot be opened!")

        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, sample_frame = cap.read()
        if not ret:
            raise IOError("Cannot read a frame from the video.")
        orig_height, orig_width = sample_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame

        # Calculate the scale factor based on target dimensions, preserving aspect ratio.
        scale_width = self.target_width / orig_width
        scale_height = self.target_height / orig_height
        scale = min(scale_width, scale_height, 1.0)  # Only downscale; if smaller, keep original.
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Determine the counting line coordinate based on orientation.
        if self.line_orientation == "horizontal":
            line_coord = int(new_height * self.line_position_ratio)
        elif self.line_orientation == "vertical":
            line_coord = int(new_width * self.line_position_ratio)
        else:
            raise ValueError("line_orientation must be 'horizontal' or 'vertical'.")

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Dynamic frame skipping.
            if frame_id % self.frame_skip != 0:
                frame_id += 1
                continue

            # Resize the frame if it's larger than the target dimensions.
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            results = self.model.track(frame, persist=True)
            annotated_frame = frame.copy()

            # Draw the counting line.
            if self.line_orientation == "horizontal":
                cv2.line(annotated_frame, (0, line_coord), (new_width, line_coord), (0, 0, 255), 4)
            else:
                cv2.line(annotated_frame, (line_coord, 0), (line_coord, new_height), (0, 0, 255), 4)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()  # [center_x, center_y, width, height]
                track_ids = results[0].boxes.id.int().cpu().tolist()
                classes = results[0].boxes.cls.cpu().tolist()

                # Iterate detected vehicles
                for box, track_id, cls in zip(boxes, track_ids, classes):
                    label = self.model.names[int(cls)]
                    if label not in self.vehicle_classes:
                        continue

                    center_x, center_y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    x1 = int(center_x - w / 2)
                    y1 = int(center_y - h / 2)
                    x2 = int(center_x + w / 2)
                    y2 = int(center_y + h / 2)

                    if self.show_video:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (230, 230, 230), 1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 230, 230), 1)

                    # Update track history
                    track = self.track_history[track_id]
                    track.append((center_x, center_y))
                    if len(track) > 30:
                        track.pop(0)
                    if len(track) >= 2 and self.show_video:
                        pts = np.array(track, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [pts], isClosed=False, color=(230, 230, 230), thickness=5)

                    speed = self.compute_speed(track, fps, self.conversion_factor,
                                               self.ref_bbox_height, h, self.frame_skip)
                    if self.show_video:
                        cv2.putText(annotated_frame, f"{speed:.1f} km/h", (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

                    if self.line_orientation == "horizontal":
                        current_side = 0 if center_y < line_coord else 1
                    else:
                        current_side = 0 if center_x < line_coord else 1

                    # Vehicle counting
                    if track_id in self.last_side:
                        if current_side != self.last_side[track_id]:
                            current_time = time.time()
                            if (track_id not in self.last_counted_time or
                                    (current_time - self.last_counted_time[track_id]) > self.cooldown_duration):
                                self.vehicle_count += 1
                                self.last_counted_time[track_id] = current_time
                                self.speeds.append(speed)
                            self.last_side[track_id] = current_side
                    else:
                        self.last_side[track_id] = current_side

            if self.show_video:
                cv2.putText(annotated_frame, f"Count: {self.vehicle_count}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Traffic Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1

        cap.release()
        if self.show_video:
            cv2.destroyAllWindows()

        avg_speed = sum(self.speeds) / len(self.speeds) if self.speeds else 0.0
        return {"vehicle_count": self.vehicle_count, "average_speed": avg_speed}
