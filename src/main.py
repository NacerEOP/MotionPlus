from ultralytics import YOLO
import cv2

# Load a pretrained YOLO model (e.g., 'yolov8n.pt' for nano, 'yolov8m.pt' for medium)
# The 'n' (nano) version is fast but less accurate; 'm' (medium) provides a good balance.
model = YOLO("yolov8n.pt")

# Run inference on an image, video, or webcam stream
# Use 'source="image.jpg"' for an image, 'source="video.mp4"' for a video,
# or 'source=0' for the default webcam.
results = model.predict(source="Algeria2B6.jpg",save=True, conf=0.5)



# The results are saved automatically in a 'runs/detect/predict' directory.
# The output will include images/videos with bounding boxes and labels drawn on them.