from datetime import datetime
from enum import Enum
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Line:
    A: Point
    B: Point

    def __init__(self, ax: float, ay: float, bx: float, by: float):
        self.A = Point(ax, ay)
        self.B = Point(bx, by)


class Resolution(Enum):
    R2160p = (3840, 2160)  # 4K
    R1440p = (2560, 1440)  # 2K
    R1080p = (1920, 1080)  # HD
    R720p = (1280, 720)  # HD
    R480p = (854, 480)  # SD
    R360p = (640, 360)  # SD
    R240p = (426, 240)  # SD
    Default = (320, 240)

    @property
    def width(self):
        return self.value[0]

    @property
    def height(self):
        return self.value[1]


@dataclass
class Video:
    traffic_cam_id: int
    video_path: str
    start_datetime: datetime
    end_datetime: datetime
    start_ref_line: Line
    finish_ref_line: Line
    ref_distance: int
    track_orientation: str = "horizontal"

    @classmethod
    def from_json(cls, data: dict):
        # Convert timestamps (float) to datetime
        start_dt = datetime.fromtimestamp(data["start_time"])
        end_dt = datetime.fromtimestamp(data["end_time"])

        # Convert to Line objects
        s = data["start_ref_line"]
        f = data["finish_ref_line"]
        start_line = Line(s["ax"], s["ay"], s["bx"], s["by"])
        finish_line = Line(f["ax"], f["ay"], f["bx"], f["by"])

        return cls(
            traffic_cam_id=data["traffic_cam_id"],
            video_path=f"downloaded_videos/{data['video_filename']}",
            start_datetime=start_dt,
            end_datetime=end_dt,
            start_ref_line=start_line,
            finish_ref_line=finish_line,
            ref_distance=int(data["ref_distance"]),
            track_orientation=data.get("track_orientation", "horizontal")
        )
