import cv2
from ultralytics import YOLO
import numpy as np
import os
from datetime import datetime
import requests
from pytz import timezone
import time

# Define constants
COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (0, 255, 255),
    3: (0, 165, 255)
}

LABELS = {
    0: "Empty Bucket",
    1: "Filled Popcorn Bucket",
    2: "Not Full Bucket",
    3: "Not Real Bucket",
}

CONFIDENCE_THRESHOLD = 0.7
API_URL = "Your API url for post"
output_folder = 'detections'  # Folder to save images locally

# Track the last detection time per class and the last empty bucket detection
last_empty_bucket = None  # To store the last empty bucket frame and timestamp
first_filled_bucket = None  # To store the first filled bucket frame and timestamp
filled_bucket_detected = False  # To track if the filled bucket has been detected after the empty bucket


def load_model(weights_path):
    """Load the YOLOv8 model."""
    try:
        print(f"Loading model from {weights_path}...")
        model = YOLO(weights_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def get_color_and_label(class_id):
    """Return the color and label based on class ID."""
    color = COLORS.get(class_id, (255, 255, 255))
    label = LABELS.get(class_id, f"Class {class_id}")
    return color, label


def send_detection_to_api(image_path, label, timestamp):
    """Send detection details (image path, label, timestamp) to the API."""
    data = {
        "detected_class": label,
        "timestamp": timestamp,
        "image_path": image_path,
    }

    try:
        response = requests.post(API_URL, data=data)
        if response.status_code == 200:
            print(f"Data for {label} sent to API successfully.")
        else:
            print(f"Failed to send data to API. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending to API: {e}")


def save_detection(frame, label, timestamp, tag):
    """
    Save a detection image with label and timestamp and send it to the API.
    """
    # Create folder for saving images
    save_folder = os.path.join(output_folder, f"{tag}_Detections")
    os.makedirs(save_folder, exist_ok=True)

    # Generate filename
    filename = f"{timestamp}_{np.random.randint(100, 999)}.jpeg"
    filepath = os.path.join(save_folder, filename)

    # Save the frame
    cv2.imwrite(filepath, frame)
    print(f"Saved {tag} detection: {filepath}")

    # Prepare the image path for the database
    image_path_for_db = f"{tag.lower()}_bucket/{filename}"

    # Send detection details to the API
    send_detection_to_api(image_path_for_db, label, timestamp)


def process_detections(frame, results):
    """Process detections and manage LIFO and FIFO behavior."""
    global last_empty_bucket, first_filled_bucket

    now = datetime.now()

    for result in results:  # Iterate over detected objects
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])  # Get class ID
            confidence = box.conf[0].item()  # Get confidence score

            if confidence >= CONFIDENCE_THRESHOLD:
                if class_id == 0:  # Empty Bucket
                    # Always update to the latest (LIFO behavior)
                    timestamp = now.strftime('%Y%m%d-%H%M%S')
                    last_empty_bucket = {
                        "timestamp": timestamp,
                        "frame": frame.copy(),
                        "class": LABELS[class_id],
                    }
                    print(f"Updated last empty bucket: {LABELS[class_id]} at {timestamp}")

                elif class_id == 1:  # Filled Bucket
                    # Capture the first detected filled bucket (FIFO behavior)
                    if first_filled_bucket is None:  # Only set the first filled bucket
                        timestamp = now.strftime('%Y%m%d-%H%M%S')
                        first_filled_bucket = {
                            "timestamp": timestamp,
                            "frame": frame.copy(),
                            "class": LABELS[class_id],
                        }
                        print(f"Captured first filled bucket: {LABELS[class_id]} at {timestamp}")

    # Save detections if both conditions are met
    if last_empty_bucket and first_filled_bucket:
        save_detection(
            last_empty_bucket["frame"], last_empty_bucket["class"], last_empty_bucket["timestamp"], "Empty"
        )
        save_detection(
            first_filled_bucket["frame"], first_filled_bucket["class"], first_filled_bucket["timestamp"], "Filled"
        )

        # Reset states after saving
        last_empty_bucket = None
        first_filled_bucket = None
        print("Resetting state for next detection cycle.")


def draw_results(frame, results):
    """Draw bounding boxes and send detection data."""
    process_detections(frame, results)  # Process detections and manage empty/fill behavior

    # Displaying the frame (optional - could also be part of process_detections if needed)
    cv2.imshow("Detection Frame", frame)


def detect_live_feed(model):
    """Capture live video feed and detect objects using YOLOv8."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting live feed...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize frame for faster processing (optional)
        resized_frame = cv2.resize(frame, (640, 480))

        # Make predictions on the resized frame
        results = model(resized_frame)

        # Draw results and send to API
        draw_results(resized_frame, results)

        # Display the frame
        cv2.imshow('YOLOv8 Live Feed', resized_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Path to the trained YOLOv8 model
    model_path = r'model\YoloV8model19112024.pt'

    # Load the model
    model = load_model(model_path)

    # Start detecting objects in the live feed
    detect_live_feed(model)
