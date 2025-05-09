from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import mimetypes
import json
from datetime import datetime
import queue

app = Flask(__name__)
CORS(app)

# SQLite configuration
db_filename = 'traffic_cams.db'
db_path = os.path.join(os.path.dirname(__file__), db_filename)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class TrafficCam(db.Model):
    __tablename__ = 'traffic_cams'
    traffic_cam_id = db.Column(db.Integer, primary_key=True)
    alias = db.Column(db.String(128), nullable=False)
    location_lat = db.Column(db.Float, nullable=False)
    location_lng = db.Column(db.Float, nullable=False)
    start_ax = db.Column(db.Float, nullable=False)
    start_ay = db.Column(db.Float, nullable=False)
    start_bx = db.Column(db.Float, nullable=False)
    start_by = db.Column(db.Float, nullable=False)
    finish_ax = db.Column(db.Float, nullable=False)
    finish_ay = db.Column(db.Float, nullable=False)
    finish_bx = db.Column(db.Float, nullable=False)
    finish_by = db.Column(db.Float, nullable=False)
    ref_distance = db.Column(db.Float, nullable=False)
    track_orientation = db.Column(db.String(16), nullable=False)

    def to_dict(self):
        return {
            'traffic_cam_id': self.traffic_cam_id,
            'alias': self.alias,
            'location_lat': self.location_lat,
            'location_lng': self.location_lng,
            'start_ref_line': {'ax': self.start_ax, 'ay': self.start_ay, 'bx': self.start_bx, 'by': self.start_by},
            'finish_ref_line': {'ax': self.finish_ax, 'ay': self.finish_ay, 'bx': self.finish_bx, 'by': self.finish_by},
            'ref_distance': self.ref_distance,
            'track_orientation': self.track_orientation
        }

# Initialize DB: create file and tables if missing
def init_db():
    # create DB file and tables within application context
    if not os.path.exists(db_path):
        with app.app_context():
            db.create_all()
            example = TrafficCam(
                traffic_cam_id=2,
                alias='Avenida Primero de Mayo Tec Madero',
                location_lat=22.254044,
                location_lng=-97.848944,
                start_ax=297, start_ay=234, start_bx=560, start_by=241,
                finish_ax=83, finish_ay=360, finish_bx=779, finish_by=395,
                ref_distance=40.0,
                track_orientation='horizontal'
            )
            db.session.add(example)
            db.session.commit()

# Call init_db at startup
init_db()


init_db()

@app.route('/cams', methods=['GET'])
def get_cams():
    cams = TrafficCam.query.all()
    return jsonify([cam.to_dict() for cam in cams])

@app.route('/cams', methods=['POST'])
def add_cam():
    data = request.get_json(force=True)
    required = ['traffic_cam_id', 'alias', 'location_lat', 'location_lng', 'start_ref_line', 'finish_ref_line', 'ref_distance', 'track_orientation']
    if not all(field in data for field in required):
        return jsonify({'error': 'Missing required field'}), 400
    cam = TrafficCam(
        traffic_cam_id=data['traffic_cam_id'],
        alias=data['alias'],
        location_lat=data['location_lat'],
        location_lng=data['location_lng'],
        start_ax=data['start_ref_line']['ax'], start_ay=data['start_ref_line']['ay'],
        start_bx=data['start_ref_line']['bx'], start_by=data['start_ref_line']['by'],
        finish_ax=data['finish_ref_line']['ax'], finish_ay=data['finish_ref_line']['ay'],
        finish_bx=data['finish_ref_line']['bx'], finish_by=data['finish_ref_line']['by'],
        ref_distance=data['ref_distance'],
        track_orientation=data['track_orientation']
    )
    db.session.add(cam)
    db.session.commit()
    return jsonify(cam.to_dict()), 201

@app.route('/cams/<int:cam_id>', methods=['PUT'])
def update_cam(cam_id):
    cam = TrafficCam.query.get(cam_id)
    if not cam:
        return jsonify({'error': 'Camera not found'}), 404
    data = request.get_json(force=True)
    if 'alias' in data:
        cam.alias = data['alias']
    if 'location_lat' in data:
        cam.location_lat = data['location_lat']
    if 'location_lng' in data:
        cam.location_lng = data['location_lng']
    if 'start_ref_line' in data:
        s = data['start_ref_line']
        cam.start_ax = s.get('ax', cam.start_ax)
        cam.start_ay = s.get('ay', cam.start_ay)
        cam.start_bx = s.get('bx', cam.start_bx)
        cam.start_by = s.get('by', cam.start_by)
    if 'finish_ref_line' in data:
        f = data['finish_ref_line']
        cam.finish_ax = f.get('ax', cam.finish_ax)
        cam.finish_ay = f.get('ay', cam.finish_ay)
        cam.finish_bx = f.get('bx', cam.finish_bx)
        cam.finish_by = f.get('by', cam.finish_by)
    if 'ref_distance' in data:
        cam.ref_distance = data['ref_distance']
    if 'track_orientation' in data:
        cam.track_orientation = data['track_orientation']
    db.session.commit()
    return jsonify(cam.to_dict())

# Video queue endpoint with DB metadata lookup
VIDEO_DIRECTORY = '/home/pedro-ayon/dev/ComputoUbicuo/Proyecto/TrafficDetection/test/video'
video_queue = queue.Queue()
# seed original entry
video_queue.put({
    "traffic_cam_id": 2,
    "video_filename": "5.avi",
    "start_datetime": "2024-03-22T14:40:15Z",
    "end_datetime":   "2024-03-22T14:45:15Z",
})

@app.route('/get_video_from_queue', methods=['GET'])
def get_video_from_queue():
    if video_queue.empty():
        return jsonify({'error': 'No videos in queue'}), 404
    video_data = video_queue.get()

    # Fetch metadata from DB
    cam = TrafficCam.query.get(video_data['traffic_cam_id'])
    if not cam:
        return jsonify({'error': 'Camera metadata not found'}), 404

    # Build combined metadata
    db_meta = cam.to_dict()
    combined_meta = {
        'traffic_cam_id': video_data['traffic_cam_id'],
        'video_filename': video_data['video_filename'],
        'start_datetime': video_data['start_datetime'],
        'end_datetime': video_data['end_datetime'],
        **{k: db_meta[k] for k in ['start_ref_line', 'finish_ref_line', 'ref_distance', 'track_orientation']}
    }

    video_filepath = os.path.join(VIDEO_DIRECTORY, combined_meta['video_filename'])
    if not os.path.exists(video_filepath):
        return jsonify({'error': 'Video file not found'}), 404

    mime_type, _ = mimetypes.guess_type(video_filepath)
    resp = send_file(video_filepath, mimetype=mime_type or 'application/octet-stream')
        # Safely format header without nesting same quotes
    filename = combined_meta['video_filename']
    resp.headers['Content-Disposition'] = f"attachment; filename=\"{filename}\""
    # Add metadata header
    resp.headers['X-Video-Metadata'] = json.dumps(combined_meta)
    return resp

# Start the Flask development server if run as script
if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
