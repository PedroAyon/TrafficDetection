from dataclasses import dataclass
from datetime import datetime

@dataclass
class Video:
    traffic_cam_id: int
    conversion_factor: float
    ref_bbox_height: int
    video_path: str
    start_datetime: datetime
    end_datetime: datetime
    line_orientation: str = "horizontal"  # default value if not provided
    line_position_ratio: float = 0.7        # default value if not provided

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            traffic_cam_id=data["traffic_cam_id"],
            conversion_factor=data["conversion_factor"],
            ref_bbox_height=data["ref_bbox_height"],
            video_path=data["video_path"],
            # Convert ISO8601 strings (with "Z") to Python datetime objects.
            start_datetime=datetime.fromisoformat(data["start_datetime"].replace("Z", "+00:00")),
            end_datetime=datetime.fromisoformat(data["end_datetime"].replace("Z", "+00:00")),
            line_orientation=data.get("line_orientation", "horizontal"),
            line_position_ratio=data.get("line_position_ratio", 0.7)
        )
