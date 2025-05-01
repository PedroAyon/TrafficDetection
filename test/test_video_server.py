from flask import Flask, jsonify, send_file
import queue
import os
import mimetypes
import json

app = Flask(__name__)

# Simulated video queue (FIFO)
video_queue = queue.Queue()

# Populate the queue with some dummy video metadata
# video_queue.put({
#     "traffic_cam_id": 1,
#     "conversion_factor": 0.15,
#     "ref_bbox_height": 40,
#     "video_filename": "4.mp4",
#     "start_datetime": "2024-03-22T14:40:15Z",
#     "end_datetime": "2024-03-22T14:45:15Z",
#     "line_orientation": "horizontal",
#     "line_position_ratio": 0.7
# })
video_queue.put({
    "traffic_cam_id": 2,
    "video_filename": "5.avi",
    "start_datetime": "2024-03-22T14:40:15Z",
    "end_datetime":   "2024-03-22T14:45:15Z",

    "start_ref_line": {
        "ax": 297,
        "ay": 234,
        "bx": 560,
        "by": 241
    },
    "finish_ref_line": {
        "ax":  83,
        "ay": 360,
        "bx": 779,
        "by": 395
    },
    "ref_distance":       40.0,
    "track_orientation": "horizontal",
})


# Simulated directory where videos are stored.
VIDEO_DIRECTORY = "/home/pedro-ayon/dev/ComputoUbicuo/Proyecto/TrafficDetection/test/video"  # Ensure this directory exists and contains the sample videos

@app.route("/get_video_from_queue", methods=["GET"])
def get_video_from_queue():
    if video_queue.empty():
        return jsonify({"error": "No videos in queue"}), 404

    video_data = video_queue.get()
    print("Serving video metadata:", video_data)
    video_filename = os.path.join(VIDEO_DIRECTORY, video_data["video_filename"])

    if not os.path.exists(video_filename):
        return jsonify({"error": "Video file not found: " + video_filename}), 404

    # Determine MIME type dynamically
    mime_type, _ = mimetypes.guess_type(video_filename)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Fallback for unknown types

    # Use send_file to stream the video file. Also, attach the metadata as a custom header.
    response = send_file(video_filename, mimetype=mime_type)
    response.headers["Content-Disposition"] = f'attachment; filename="{video_data["video_filename"]}"'
    response.headers["X-Video-Metadata"] = json.dumps(video_data)
    return response

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
