import datetime
from collections import defaultdict

import cv2

from models import Line


class TrafficDetector:
    def __init__(self, video_path: str, model, start_ref_line: Line, finish_ref_line: Line, ref_distance: int,
                 track_orientation: str, show_video: bool = False, cooldown_duration: float = 2.0,
                 frame_skip: int = 1, vehicle_classes: set = None, target_width: int = 1280,
                 target_height: int = 720):
        self.video_path = video_path
        self.model = model
        self.start_ref_line = start_ref_line
        self.finish_ref_line = finish_ref_line
        self.ref_distance = ref_distance
        self.track_orientation = track_orientation
        self.show_video = show_video
        self.cooldown_duration = cooldown_duration
        self.vehicle_classes = vehicle_classes or {"car", "truck", "bus", "motorcycle", "van"}
        self.frame_skip = frame_skip
        self.target_width = target_width
        self.target_height = target_height
        self.track_history = defaultdict(list)
        # store timestamps for start-first or finish-first crossings
        self.start_times = {}
        self.finish_times = {}
        self.vehicle_count = 0
        self.speeds = []

    @staticmethod
    def compute_speed(start_time: datetime.datetime, finish_time: datetime.datetime,
                      ref_distance: float) -> float:
        time_diff = (finish_time - start_time).total_seconds()
        if time_diff <= 0:
            return 0.0
        speed_m_per_s = ref_distance / time_diff
        return speed_m_per_s * 3.6

    @staticmethod
    def has_vehicle_crossed_line(ref_line: Line, track: list) -> bool:
        if len(track) < 2:
            return False
        p1 = track[-2]
        p2 = track[-1]
        ax, ay = ref_line.A.x, ref_line.A.y
        bx, by = ref_line.B.x, ref_line.B.y
        # cross product sign
        val1 = (bx - ax) * (p1[1] - ay) - (by - ay) * (p1[0] - ax)
        val2 = (bx - ax) * (p2[1] - ay) - (by - ay) * (p2[0] - ax)
        return val1 * val2 < 0

    def process_video(self) -> dict:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError("Video file not found or cannot be opened!")

        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, sample_frame = cap.read()
        if not ret:
            raise IOError("Cannot read a frame from the video.")
        orig_h, orig_w = sample_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        scale = min(self.target_width / orig_w, self.target_height / orig_h, 1.0)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        video_start_time = datetime.datetime.now()
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % self.frame_skip != 0:
                frame_id += 1
                continue
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            results = self.model.track(frame, persist=True)
            annotated = frame.copy()

            # draw start (green) and finish (red) lines
            cv2.line(annotated,
                     (int(self.start_ref_line.A.x), int(self.start_ref_line.A.y)),
                     (int(self.start_ref_line.B.x), int(self.start_ref_line.B.y)),
                     (0, 255, 0), 2)
            cv2.line(annotated,
                     (int(self.finish_ref_line.A.x), int(self.finish_ref_line.A.y)),
                     (int(self.finish_ref_line.B.x), int(self.finish_ref_line.B.y)),
                     (0, 0, 255), 2)

            current_time = video_start_time + datetime.timedelta(seconds=frame_id / fps)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()  # center_x, center_y, w, h
                ids = results[0].boxes.id.int().cpu().tolist()
                classes = results[0].boxes.cls.cpu().tolist()

                for box, track_id, cls in zip(boxes, ids, classes):
                    label = self.model.names[int(cls)]
                    if label not in self.vehicle_classes:
                        continue

                    cx, cy, w, h = map(float, box)
                    # compute corners
                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)

                    if self.show_video:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (230, 230, 230), 1)
                        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (230, 230, 230), 2)

                    # update track history
                    hist = self.track_history[track_id]
                    hist.append((cx, cy))
                    if len(hist) > 30:
                        hist.pop(0)

                    crossed_start = self.has_vehicle_crossed_line(self.start_ref_line, hist)
                    crossed_finish = self.has_vehicle_crossed_line(self.finish_ref_line, hist)

                    # when crossing start line
                    if crossed_start:
                        if track_id in self.finish_times:
                            finish_t = self.finish_times.pop(track_id)
                            speed = self.compute_speed(current_time, finish_t, self.ref_distance)
                            self.speeds.append(speed)
                        elif track_id not in self.start_times:
                            self.start_times[track_id] = current_time

                    # when crossing finish line, count always
                    if crossed_finish:
                        self.vehicle_count += 1
                        if track_id in self.start_times:
                            start_t = self.start_times.pop(track_id)
                            speed = self.compute_speed(start_t, current_time, self.ref_distance)
                            self.speeds.append(speed)
                        elif track_id not in self.finish_times:
                            self.finish_times[track_id] = current_time

            frame_id += 1
            if self.show_video:
                cv2.imshow("Traffic", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if self.show_video:
            cv2.destroyAllWindows()

        avg_speed = sum(self.speeds) / len(self.speeds) if self.speeds else 0.0
        return {"vehicle_count": self.vehicle_count, "average_speed": avg_speed}
