import aiohttp
import time
from models import Video

DATA_SERVER_REGISTER_ENDPOINT_URL = "https://insect-promoted-gnu.ngrok-free.app/record"


async def send_to_data_server(video_obj: Video, result: dict):
    payload = {
        "traffic_cam_id": video_obj.traffic_cam_id,
        "start_datetime": video_obj.start_datetime.isoformat(),
        "end_datetime": video_obj.end_datetime.isoformat(),
        "vehicle_count": result.get("vehicle_count"),
        "average_speed": float(result.get("average_speed", 0))
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(DATA_SERVER_REGISTER_ENDPOINT_URL, json=payload) as resp:
                if resp.status == 200:
                    print(f"[{time.strftime('%X')}] Successfully sent data: {payload}")
                else:
                    print(f"[{time.strftime('%X')}] Failed to send data, status: {resp.status}")
    except Exception as e:
        print(f"Exception sending data to server: {e}")