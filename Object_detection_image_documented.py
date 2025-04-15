from unicodedata import category  # Imports the 'category' function from the 'unicodedata' module (though not used in the current code).

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import cv2

# ------------------------- OBJECT DETECTOR CONFIGURATION -------------------------
# Specify the configuration options for the MediaPipe object detector.
options = vision.ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path="/Python/Object_Deteccion/Models/efficientdet_lite_float32.tflite"),
    max_results=5,  # Sets the maximum number of detected objects to display (top 5 with highest confidence).
    score_threshold=0.2,  # Defines the confidence threshold. Only objects with a score above 0.2 will be considered.
    running_mode=vision.RunningMode.IMAGE  # Specifies that the detector will be used to process static images.
)

# Creates an instance of the object detector from the configuration options.
detector = vision.ObjectDetector.create_from_options(options)

# ------------------------- IMAGE LOADING AND PREPARATION -------------------------
# Read the image using OpenCV's 'imread' function.
image = cv2.imread("./Data/image.jpg")
# Convert the image from BGR color space (default in OpenCV) to RGB (required by MediaPipe).
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Create a MediaPipe image object from the RGB image.
image_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

# ------------------------- OBJECT DETECTION -------------------------
# Perform object detection on the image using the created detector.
detection_result = detector.detect(image_rgb)

# ------------------------- PROCESSING AND VISUALIZING RESULTS -------------------------
# Iterate through each detected object found in the image.
for detection in detection_result.detections:
    # Print the complete information of the detection (bounding box, categories, scores, etc.).
    print(detection)

    # Get the coordinates of the bounding box of the detected object.
    bbox = detection.bounding_box
    bbox_x, bbox_y, bbox_w, bbox_h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

    # Get the category information (name and score) of the detected object.
    category = detection.categories[0]  # Each detection can have multiple categories, but the first one (with the highest confidence) is usually taken.
    score = category.score * 100  # Get the confidence score and convert it to a percentage.
    category_name = category.category_name  # Get the name of the object's category.

    # Draw a background rectangle for the object label above the bounding box.
    cv2.rectangle(image, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y - 30), (100, 255, 0), -1)  # -1 fills the rectangle.
    # Draw the bounding box around the detected object on the original image.
    cv2.rectangle(image, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (100, 255, 0), 2)  # 2 indicates the border thickness.
    # Put the text with the category name and confidence score above the bounding box.
    cv2.putText(image, f"{category_name}: {score:.2f}%", (bbox_x + 5, bbox_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# ------------------------- FINAL IMAGE VISUALIZATION -------------------------
# Display the image with the detected objects and their bounding boxes and labels.
cv2.imshow("Image", image)
# Wait for a key press to close the image window.
cv2.waitKey(0)
# Destroy all OpenCV windows.
cv2.destroyAllWindows()