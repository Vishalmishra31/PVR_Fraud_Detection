import cv2
import numpy as np
import os
import pathlib
import threading
import queue
import time
from ultralytics import YOLO
from datetime import datetime

# Modify pathlib for compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Colors for class IDs
COLORS = {
    0: (0, 0, 255),  # Class 0: Red (Empty Bucket)
    1: (0, 255, 0),  # Class 1: Green (Filled Popcorn Bucket)
    2: (0, 255, 255),  # Class 2: Orange (Not Full Bucket)
    3: (0, 165, 255)   # Class 3: Yellow (Not Real Bucket)
}

# Labels for class IDs
LABELS = {
    0: "Empty Bucket",
    1: "Filled Popcorn Bucket",
    2: "Not Full Bucket",
    3: "Not Real Bucket",
}

CONFIDENCE_THRESHOLD = 0.7  # Adjusted confidence threshold

# Folder paths for detected and comparison images
output_folder = r'Detected Objects'
comparison_folder = r'Comparison Folder'
matched_folder = r'Complete Matches'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(comparison_folder, exist_ok=True)
os.makedirs(matched_folder, exist_ok=True)

# Frame and processing queues for multiple cameras
frame_queues = {}
processed_queues = {}

def load_model(weights_path):
    """Load YOLOv8 model with exception handling."""
    try:
        print(f"Loading model from {weights_path}...")
        model = YOLO(weights_path)  # Load YOLOv8 model
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def get_color_and_label(class_id):
    """Map class IDs to colors and labels."""
    color = COLORS.get(class_id, (255, 255, 255))  # Default color: White
    label = LABELS.get(class_id, f"Class {class_id}")  # Default label
    return color, label

def save_detected_object(frame, box, label, feed_id):
    """Save the detected object as an image in the output folder with a feed identifier."""
    x1, y1, x2, y2 = map(int, box[:4])
    cropped_image = frame[y1:y2, x1:x2]

    # Create a unique file name using timestamp and feed_id
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{feed_id}_{label}_{timestamp}.jpg"
    filepath = os.path.join(output_folder, filename)

    # Save the cropped image
    cv2.imwrite(filepath, cropped_image)
    print(f"Saved detected object from feed {feed_id}: {filepath}")

    # Check for matching images in the comparison folder
    match_images(filepath, timestamp)

def match_images(detected_path, timestamp):
    """Check for a matching image in the comparison folder based on timestamp and move to matched folder if found."""
    for filename in os.listdir(comparison_folder):
        if timestamp in filename:
            comparison_path = os.path.join(comparison_folder, filename)
            
            # Move both images to the matched folder
            matched_detected_path = os.path.join(matched_folder, f"Detected_{timestamp}.jpg")
            matched_comparison_path = os.path.join(matched_folder, f"Comparison_{timestamp}.jpg")
            
            os.rename(detected_path, matched_detected_path)
            os.rename(comparison_path, matched_comparison_path)
            print(f"Matched and moved: {matched_detected_path} and {matched_comparison_path}")

            # Display images side-by-side
            show_images_side_by_side(matched_detected_path, matched_comparison_path, timestamp)
            break

def show_images_side_by_side(img_path1, img_path2, timestamp):
    """Display two images side-by-side with their timestamp."""
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    # Resize images for consistency
    img1 = cv2.resize(img1, (640, 480))
    img2 = cv2.resize(img2, (640, 480))

    # Concatenate images horizontally
    combined_img = np.hstack((img1, img2))

    # Add timestamp as text on the combined image
    cv2.putText(combined_img, timestamp, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the combined image
    cv2.imshow('Complete Match', combined_img)
    cv2.waitKey(3000)  # Display for 3 seconds

def draw_results(frame, results, feed_id):
    """Draw bounding boxes and labels on the frame."""
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])  # Get class ID
            confidence = box.conf[0].item()  # Get confidence score

            # Filter by class ID and confidence threshold
            if class_id in LABELS and confidence >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                color, label = get_color_and_label(class_id)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{label} {confidence:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Save the detected object
                save_detected_object(frame, box.xyxy[0], label, feed_id)

def frame_reader(cap, feed_id):
    """Function to read frames from the RTSP stream in a separate thread for each feed."""
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))  # Resize for consistency
            if not frame_queues[feed_id].full():
                frame_queues[feed_id].put(frame)
        time.sleep(0.03)  # Limit frame rate (e.g., 30 FPS)

def frame_processor(model, feed_id):
    """Function to process frames from the queue in a separate thread for each feed."""
    while True:
        if not frame_queues[feed_id].empty():
            frame = frame_queues[feed_id].get()
            results = model(frame)  # Perform inference on the frame
            draw_results(frame, results, feed_id)  # Draw results on the frame
            if not processed_queues[feed_id].full():
                processed_queues[feed_id].put(frame)

def start_feed_processing(feed_id, rtsp_url, model):
    """Initialize the feed processing for a single RTSP feed."""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream for feed {feed_id}.")
        return

    frame_queues[feed_id] = queue.Queue(maxsize=10)
    processed_queues[feed_id] = queue.Queue(maxsize=10)

    threading.Thread(target=frame_reader, args=(cap, feed_id), daemon=True).start()
    threading.Thread(target=frame_processor, args=(model, feed_id), daemon=True).start()

def main(weights_path, rtsp_urls):
    """Main function to capture video from multiple RTSP feeds and display results."""
    model = load_model(weights_path)

    # Start a processing thread for each RTSP feed
    for feed_id, rtsp_url in enumerate(rtsp_urls):
        print(f"Starting feed {feed_id} for URL: {rtsp_url}")
        start_feed_processing(feed_id, rtsp_url, model)

    while True:
        for feed_id in range(len(rtsp_urls)):
            if not processed_queues[feed_id].empty():
                frame = processed_queues[feed_id].get()
                cv2.imshow(f'YOLOv8 Detection - Feed {feed_id}', frame)

            # Check if any key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    weights_path = r'model\YoloV8model19112024.pt'  # Update with your YOLOv8 model path
    rtsp_urls = [
        'rtsp://UserName:Password@IpAddress/cam/realmonitor?channel=3&subtype=0',
        'rtsp://UserName:Password@IpAddress/cam/realmonitor?channel=4&subtype=0',
        
    ]
    main(weights_path, rtsp_urls)
