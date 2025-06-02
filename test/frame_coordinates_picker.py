import cv2

# Constants
TARGET_SIZE = (320, 240)
SCALE = 3  # display scale factor

# Function to show mouse coordinates
def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        display = param["display"].copy()
        h, w, _ = display.shape
        orig_w, orig_h = w // SCALE, h // SCALE
        scaled_x, scaled_y = x // SCALE, y // SCALE
        text = f"X: {scaled_x} / {orig_w}, Y: {scaled_y} / {orig_h}"
        cv2.putText(display, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("First Frame", display)

# Load video
video_path = "video.mp4"  # Replace with your path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read frame.")
    exit()

# Resize frame to target resolution
frame = cv2.resize(frame, TARGET_SIZE)

# Upscale for display
display_frame = cv2.resize(frame, (TARGET_SIZE[0]*SCALE, TARGET_SIZE[1]*SCALE))

# Create window and show frame
cv2.namedWindow("First Frame")
cv2.setMouseCallback("First Frame", show_coordinates, param={"display": display_frame})
cv2.imshow("First Frame", display_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
