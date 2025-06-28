import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("C:/Users/TUF GAMING/Downloads/Sign_language_data1/model/best.pt")  # Replace with the path to your model

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use '0' for the default webcam

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break
    
    # Perform prediction
    results = model(frame)
    
    # Draw predictions on the frame
    annotated_frame = results[0].plot()  # YOLOv8 automatically provides annotated frames
    
    # Display the resulting frame
    cv2.imshow('YOLOv8 Real-Time Sign Language Detection', annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
