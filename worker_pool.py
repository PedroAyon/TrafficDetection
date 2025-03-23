import asyncio
import concurrent.futures
from ai_model import AIModelFactory
from data_storage import send_to_data_server
from traffic_detector import TrafficDetector
from video import Video
from resolutions import Resolution


async def video_worker(worker_id: int, video_queue: asyncio.Queue, executor: concurrent.futures.Executor) -> None:
    """
    Worker coroutine that continuously processes videos from the queue.

    Creates its own YOLO model instance and waits for Video objects to process.
    After processing, sends the result via an HTTP PUT request.

    Args:
        worker_id: The ID of the worker.
        video_queue: The asyncio queue containing Video objects.
        executor: The ThreadPoolExecutor to run blocking video processing tasks.
    """
    loop = asyncio.get_running_loop()
    local_model = AIModelFactory.create_model("yolo", "yolo11n.pt")
    while True:
        video_obj: Video = await video_queue.get()
        try:
            print(
                f"Worker {worker_id}: Received video from camera {video_obj.traffic_cam_id} with file '{video_obj.video_path}'")
            detector = TrafficDetector(
                video_path=video_obj.video_path,
                model=local_model,
                conversion_factor=video_obj.conversion_factor,
                ref_bbox_height=video_obj.ref_bbox_height,
                line_orientation=video_obj.line_orientation,
                line_position_ratio=video_obj.line_position_ratio,
                show_video=False,
                cooldown_duration=2.0,
                vehicle_classes={"car", "truck", "bus", "motorcycle", "van"},
                frame_skip=2,
                target_width=Resolution.R480p.width,
                target_height=Resolution.R480p.height
            )
            result = await loop.run_in_executor(executor, detector.process_video)
            print(f"Worker {worker_id}: Finished processing video from camera {video_obj.traffic_cam_id}")
            await send_to_data_server(video_obj, result)
        except Exception as e:
            print(f"Worker {worker_id}: Encountered an error: {e}")
        finally:
            video_queue.task_done()


class VideoProcessor:
    """
    Manages a pool of asynchronous workers to process videos.

    Attributes:
        num_workers: The number of workers to run.
        video_queue: An asyncio queue for incoming Video objects.
        executor: A ThreadPoolExecutor for running blocking video processing tasks.
        workers: List of worker tasks.
    """

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.video_queue = asyncio.Queue()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self.workers = []
        self._started = False

    async def start(self):
        """
        Starts the worker tasks if they haven't been started already.
        """
        if not self._started:
            self.workers = [
                asyncio.create_task(video_worker(worker_id=i, video_queue=self.video_queue, executor=self.executor))
                for i in range(self.num_workers)
            ]
            self._started = True

    async def add_video(self, video_obj: Video):
        """
        Adds a Video object to the processing queue.

        Args:
            video_obj: A Video instance containing video metadata and file path.
        """
        await self.video_queue.put(video_obj)
        print(f"VideoProcessor: Enqueued video from camera {video_obj.traffic_cam_id}")

    async def stop(self):
        """
        Stops all worker tasks after processing all videos in the queue.
        """
        await self.video_queue.join()
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.executor.shutdown(wait=True)
