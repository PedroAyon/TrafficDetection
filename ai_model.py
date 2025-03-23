import io
import logging
import os
import sys

import torch
from ultralytics import YOLO

# 1. Disable all Ultralytics/YOLO logging FIRST
os.environ["YOLO_VERBOSE"] = "False"  # Environment variable (if supported)
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)  # Disable all logging


# 2. Create a "nuclear option" context manager
class CompleteSilence:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        # Also suppress low-level CUDA messages
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        torch.set_printoptions(profile="default")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class AIModelFactory:
    @staticmethod
    def create_model(model_type: str, model_path: str):
        if model_type.lower() == "yolo":
            with CompleteSilence():  # Use our custom suppressor
                # Force-disable CUDA logging
                torch.backends.cudnn.benchmark = False
                torch.set_warn_always(False)

                # Load model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = YOLO(model_path).to(device)

                return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")