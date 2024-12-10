# PVR_Fraud_Detection

# **Bucket Detection System using YOLO**

This project leverages the **YOLO (You Only Look Once)** object detection model to detect and classify buckets as either **Empty** or **Filled Popcorn Buckets** from a live webcam feed. The system integrates with an API to save detection details, including the detected image, timestamp, and a hardcoded camera ID (`CAM1`).

---

## **Features**

- **Real-time Detection**:
  - Detects and classifies **Empty Buckets** and **Filled Popcorn Buckets**.
  - Bounding boxes are drawn around detected objects with confidence scores.

- **Configurable Detection**:
  - **Cooldown Timers**:
    - Empty Buckets: 5 seconds
    - Filled Buckets: 2 seconds
    - If an Empty Bucket is detected, a Filled Bucket can also be detected within the cooldown period.

- **API Integration**:
  - Sends detection data to a REST API.
  - Data includes:
    - Detected class (`Empty Bucket` or `Filled Popcorn Bucket`)
    - Timestamp in IST format
    - Hardcoded `camera_id` as `CAM1`
    - Detected image file

- **Easy-to-Understand Visualization**:
  - Real-time display of the webcam feed with labeled bounding boxes.

---

## **Technologies Used**

- **Programming Language**: Python
- **Object Detection**: YOLO (Ultralytics)
- **Computer Vision**: OpenCV
- **API Communication**: Python `requests` library
- **Date/Time Management**: Python `datetime` and `pytz`

---

## **Installation**

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/<your-username>/bucket-detection-system.git
    cd bucket-detection-system
    ```

2. **Install Dependencies**:
    Make sure Python 3.8+ is installed. Install the required libraries:
    ```bash
    pip install ultralytics opencv-python-headless requests pytz
    ```

3. **Download YOLO Weights**:
    - Place your trained YOLO weights file in the `model/` directory.
    - Update the `weights_path` in the script:
      ```python
      weights_path = r"model\best1656.pt"
      ```

4. **Setup Output Directory**:
    The script automatically creates an `Detected Objects/` directory to save detection images.

---

## **Usage**

1. **Run the Script**:
    ```bash
    python bucket_detection.py
    ```

2. **Detection Workflow**:
    - The system starts a live webcam feed.
    - Detected objects are labeled in real-time on the video frame.
    - Detected images and metadata are sent to the configured API endpoint (`https://trizoapidev.tsoft.co.in/api/pvr-images/`).

3. **Stop the Program**:
    - Press the `q` key to exit the webcam feed.

---

## **Project Structure**

```
bucket-detection-system/
├── bucket_detection.py    # Main script for detection
├── model/                 # Directory to store YOLO weights
│   └── best1656.pt        # Trained YOLO weights
├── Detected Objects/      # Directory for saved detection images
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

---

## **API Details**

The system integrates with a REST API to upload detection details. The API expects the following fields:

| **Field**        | **Type**   | **Description**                         |
|-------------------|------------|-----------------------------------------|
| `detected_class` | `string`   | The detected class (`Empty Bucket` or `Filled Popcorn Bucket`). |
| `timestamp`      | `datetime` | Detection timestamp in IST format.       |
| `camera_id`      | `string`   | Hardcoded value: `CAM1`.                 |
| `image`          | `file`     | JPEG image of the detected object.       |

---

## **Configuration**

You can adjust the following parameters in the script:

- **Confidence Threshold**:
  - Modify `CONFIDENCE_THRESHOLD` to adjust the minimum confidence for detections.

- **Cooldown Timers**:
  - Modify `COOLDOWNS` for custom cooldown durations:
    ```python
    COOLDOWNS = {0: timedelta(seconds=5), 1: timedelta(seconds=2)}
    ```

- **API Endpoint**:
  - Update `API_URL` for a different endpoint:
    ```python
    API_URL = "https://your-api-url/api-endpoint/"
    ```

---

## **Known Limitations**

- Requires a stable webcam connection.
- Ensure the API endpoint is accessible and responds correctly.

---

## **Future Enhancements**

- Add support for detecting additional bucket types.
- Extend functionality to process recorded videos instead of just live feeds.
- Improve API error handling and logging.

---

## **License**

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

---

Feel free to reach out or contribute to the project! 😊


