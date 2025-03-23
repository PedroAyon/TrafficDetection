from enum import Enum

class Resolution(Enum):
    R2160p = (3840, 2160)  # 4K
    R1440p = (2560, 1440)  # 2K
    R1080p = (1920, 1080)  # HD
    R720p  = (1280, 720)   # HD
    R480p  = (854, 480)    # SD
    R360p  = (640, 360)    # SD
    R240p  = (426, 240)    # SD

    @property
    def width(self):
        return self.value[0]

    @property
    def height(self):
        return self.value[1]
