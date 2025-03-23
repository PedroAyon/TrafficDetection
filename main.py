import asyncio
import aiohttp
import os
import json

from worker_pool import VideoProcessor
from video import Video

API_URL = "http://localhost:5000/get_video_from_queue"

DOWNLOAD_FOLDER = "downloaded_videos"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

async def fetch_video(session: aiohttp.ClientSession):
    """
    Calls the API endpoint to get a video.
    Expects that the API returns the video file in the response body,
    and includes the metadata as a JSON string in the 'X-Video-Metadata' header.
    """
    try:
        async with session.get(API_URL) as response:
            if response.status != 200:
                print(f"API responded with status {response.status}")
                return None, None

            # Retrieve metadata from the header.
            metadata_str = response.headers.get("X-Video-Metadata")
            if not metadata_str:
                print("No metadata header found in API response.")
                return None, None
            metadata = json.loads(metadata_str)

            # Read the video file (binary data) from the response.
            video_data = await response.read()
            return metadata, video_data

    except Exception as e:
        print(f"Exception during fetch_video: {e}")
        return None, None

async def main():
    processor = VideoProcessor(num_workers=4)
    await processor.start()
    sleep_time = 5
    async with aiohttp.ClientSession() as session:
        # Infinite loop to poll the API every <sleep_time> seconds.
        while True:
            metadata, video_data = await fetch_video(session)
            if metadata and video_data:
                video_path = os.path.join(DOWNLOAD_FOLDER, metadata["video_filename"])
                with open(video_path, "wb") as f:
                    f.write(video_data)
                print(f"Downloaded video to {video_path}")
                metadata["video_path"] = video_path
                video_obj = Video.from_json(metadata)
                await processor.add_video(video_obj)
            else:
                print("No video fetched from API.")

            await asyncio.sleep(sleep_time)

if __name__ == "__main__":
    asyncio.run(main())
