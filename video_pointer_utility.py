import cv2
import argparse
import time

# Global variable to store mouse position
mouse_x, mouse_y = -1, -1


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


def main(video_path: str, speed_factor: float = 1.0):
    global mouse_x, mouse_y

    # Desired resolution for resizing
    target_width, target_height = 854, 480

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return

    # Get original FPS to control playback speed
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(int((1000 / fps) * speed_factor), 1)

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to 480p
        frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # Overlay coordinates
        if mouse_x >= 0 and mouse_y >= 0:
            text = f"X: {mouse_x}, Y: {mouse_y}"
            cv2.putText(frame_resized, text, (10, target_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Frame', frame_resized)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC to quit
            break
        elif key == ord('f'):  # faster playback
            speed_factor = max(0.1, speed_factor - 0.1)
            delay = max(int((1000 / fps) * speed_factor), 1)
            print(f"Speed factor: {speed_factor:.1f}x")
        elif key == ord('s'):  # slower playback
            speed_factor += 0.1
            delay = max(int((1000 / fps) * speed_factor), 1)
            print(f"Speed factor: {speed_factor:.1f}x")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Utility to display video frame coordinates under mouse pointer (resized to 480p)')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('-sf', '--speed_factor', type=float, default=1.0,
                        help='Playback speed factor: >1 slower, <1 faster (default=1.0)')
    args = parser.parse_args()
    main(args.video_path, args.speed_factor)
