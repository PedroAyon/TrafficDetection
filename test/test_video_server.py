from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import cv2
import threading
import time
import os
import urllib.request
import numpy as np
import json
import mimetypes
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- Configuración SQLite ---
db_filename = 'traffic_cams.db'
db_path = os.path.join(os.path.dirname(__file__), db_filename)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# --- Modelo para cámaras ---
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


# --- Modelo para videos ---
class VideoMetadata(db.Model):
    __tablename__ = 'video_metadata'
    id = db.Column(db.Integer, primary_key=True)
    video_filename = db.Column(db.String(256), nullable=False, unique=True)
    start_time = db.Column(db.Float, nullable=False)
    end_time = db.Column(db.Float, nullable=False)
    traffic_cam_id = db.Column(db.Integer, db.ForeignKey('traffic_cams.traffic_cam_id'), nullable=True)

    def to_dict(self):
        return {
            'video_filename': self.video_filename,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.end_time - self.start_time,
            'traffic_cam_id': self.traffic_cam_id
        }


# --- Inicialización DB ---
def init_db():
    if not os.path.exists(db_path):
        with app.app_context():
            db.create_all()
            # Insertar cámaras de ejemplo
            example2 = TrafficCam(
                traffic_cam_id=1,
                alias='Calle Francia 809',
                location_lat=22.254044,
                location_lng=-97.848944,
                start_ax=297, start_ay=234, start_bx=560, start_by=241,
                finish_ax=83, finish_ay=360, finish_bx=779, finish_by=395,
                ref_distance=40.0,
                track_orientation='horizontal'
            )
            db.session.add(example2)

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


# --- Variables globales ---
output_dir = "videos"
os.makedirs(output_dir, exist_ok=True)

ESP32_CAM_STREAM_URL = "https://unlikely-above-sunbeam.ngrok-free.app/stream"  # URL de Ngrok


# --- Captura y guardado de video ---
def capture_stream():
    while True:
        cap = cv2.VideoCapture(ESP32_CAM_STREAM_URL)
        if not cap.isOpened():
            print("No se pudo abrir el stream, reintentando en 5 s...")
            time.sleep(5)
            continue

        video_start_time = None
        out = None
        TARGET_DURATION = 30.0  # segundos reales
        FIXED_FPS = 15.0  # ajusta al fps aproximado de tu ESP32-CAM

        while True:
            ret, frame = cap.read()
            if not ret:
                # Si falla la lectura, salimos para reconectar
                print("Fallo al leer frame, reiniciando captura...")
                break

            now = time.time()
            if video_start_time is None:
                # Primer frame: inicializo VideoWriter
                video_start_time = now
                start_dt = datetime.fromtimestamp(video_start_time)
                video_name = f"{output_dir}/video_{int(video_start_time)}.avi"
                print(f"\nIniciando grabación: {video_name}")
                print(f"Hora de inicio: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                h, w = frame.shape[:2]
                out = cv2.VideoWriter(video_name, fourcc, FIXED_FPS, (w, h))

            # Escribo el frame al momento
            out.write(frame)
            print(".", end="", flush=True)

            # Si llevo 30 s, cierro y guardo metadata
            elapsed = now - video_start_time
            if elapsed >= TARGET_DURATION:
                out.release()
                end_dt = datetime.fromtimestamp(now)
                print(f"\nVideo guardado: {video_name}")
                print(f"Hora de inicio: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Hora de fin:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Duración captura: {elapsed:.2f} s (FPS usado: {FIXED_FPS:.2f})")

                # Inserto metadata en la BD
                with app.app_context():
                    meta = VideoMetadata(
                        video_filename=os.path.basename(video_name),
                        start_time=video_start_time,
                        end_time=now,
                        traffic_cam_id=2
                    )
                    db.session.add(meta)
                    db.session.commit()

                break  # Termina este video y reconecta para el siguiente

        # Libero recursos y espero un instante antes de reconectar
        if out is not None:
            out.release()
        cap.release()
        time.sleep(1)


# --- Endpoints para cámaras (mantener intactos) ---
@app.route('/cams', methods=['GET'])
def get_cams():
    cams = TrafficCam.query.all()
    return jsonify([cam.to_dict() for cam in cams])


@app.route('/cams', methods=['POST'])
def add_cam():
    data = request.get_json(force=True)
    required = ['traffic_cam_id', 'alias', 'location_lat', 'location_lng', 'start_ref_line', 'finish_ref_line',
                'ref_distance', 'track_orientation']
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


# --- Endpoint para servir videos con metadata desde DB ---
@app.route('/videos', methods=['GET'])
def get_video():
    # Obtener el video más antiguo no servido (orden por start_time)
    video_meta = VideoMetadata.query.order_by(VideoMetadata.start_time).first()
    if not video_meta:
        return jsonify({"message": "No hay videos disponibles"}), 404

    video_filepath = os.path.join(output_dir, video_meta.video_filename)
    if not os.path.exists(video_filepath):
        # Si el archivo no existe, eliminar registro y devolver error
        db.session.delete(video_meta)
        db.session.commit()
        return jsonify({"message": "Archivo de video no encontrado"}), 404

    # Obtener metadata de la cámara asociada
    cam = None
    if video_meta.traffic_cam_id:
        cam = TrafficCam.query.get(video_meta.traffic_cam_id)

    # Construir metadata combinada
    combined_meta = video_meta.to_dict()
    if cam:
        cam_meta = cam.to_dict()
        combined_meta.update({
            'alias': cam_meta.get('alias'),
            'location_lat': cam_meta.get('location_lat'),
            'location_lng': cam_meta.get('location_lng'),
            'start_ref_line': cam_meta.get('start_ref_line'),
            'finish_ref_line': cam_meta.get('finish_ref_line'),
            'ref_distance': cam_meta.get('ref_distance'),
            'track_orientation': cam_meta.get('track_orientation')
        })

    mime_type, _ = mimetypes.guess_type(video_filepath)
    resp = send_file(video_filepath, mimetype=mime_type or 'application/octet-stream', as_attachment=True)
    resp.headers['X-Video-Metadata'] = json.dumps(combined_meta)

    # Eliminar el video de la cola (registro) para no servirlo de nuevo
    db.session.delete(video_meta)
    db.session.commit()

    return resp


if __name__ == "__main__":
    init_db()
    video_thread = threading.Thread(target=capture_stream, daemon=True)
    video_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=True)