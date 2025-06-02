import cv2
import time
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

def gen_frames():
    """
    Lee un archivo de vídeo ('video.mp4') y lo emite a 15 fps,
    manteniendo la velocidad de reproducción original.
    """
    cap = cv2.VideoCapture('video.mp4')
    if not cap.isOpened():
        # Si no abre el vídeo, emite un fondo negro constante
        print("WARNING: No pudo abrir 'video.mp4'. Emisión de negro.")
        blank = (0 * np.ones((240, 320, 3), dtype=np.uint8))
        ret, buf = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        black_jpg = buf.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + black_jpg + b'\r\n')

    # Obtener fps original del vídeo; si es 0 o inválido, asumimos 30 fps.
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0

    target_fps = 15.0
    frame_ratio = src_fps / target_fps  # ej: si src=30, ratio=2

    # Inicializamos contadores y tiempo de inicio
    frame_count = 0         # cuadros leídos desde el principio (0-based)
    emitted_frames = 0      # cuántos cuadros hemos emitido
    start_time = time.time()

    # Forzamos resolución menor para lectura más ligera (opcional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        success, frame = cap.read()
        if not success:
            # Si se termina el vídeo, reiniciamos contadores y volvemos al inicio
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            emitted_frames = 0
            start_time = time.time()
            continue

        frame_count += 1

        # Decidir si este cuadro debe emitirse:
        # Emitir cuando frame_count >= (emitted_frames + 1) * frame_ratio
        if frame_count < (emitted_frames + 1) * frame_ratio:
            # Si no alcanza el umbral, saltamos este frame
            continue

        # Esperar hasta que sea el momento real de emitir
        ideal_emit_time = start_time + (emitted_frames / target_fps)
        now = time.time()
        if now < ideal_emit_time:
            time.sleep(ideal_emit_time - now)

        # Codificar y enviar este "frame a 15 fps"
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            # Si falla la codificación, no contamos como emitido y seguimos
            continue

        jpg_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')

        emitted_frames += 1

    cap.release()


@app.route('/stream')
def stream():
    """
    Endpoint MJPEG: multipart/x-mixed-replace; boundary=frame
    """
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    # Flask escuchando en 0.0.0.0:5000
    app.run(host='0.0.0.0', port=5000, threaded=True)
