import cv2
import numpy as np
import csv

# Initialize the Kalman Filter parameters
kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurement variables (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)  # Measurement matrix
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)  # Transition matrix
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03  # Process noise covariance

# Read the video file or camera stream
cap = cv2.VideoCapture('video.mp4')  # Or use 0 for live camera feed

# Get the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video file or camera feed")
    exit()

# Get video properties for saving the output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for the output video
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))  # Output video name and properties

# Resize frame for better viewing during selection
resized_frame = cv2.resize(frame, (800, 600))  # Adjust the size as needed

# Select the object to track
x, y, w, h = cv2.selectROI("Select Object to Track", resized_frame, False, False)
x *= (width / 800)  # Scale x-coordinate back to original size
y *= (height / 600)  # Scale y-coordinate back to original size
w *= (width / 800)  # Scale width back to original size
h *= (height / 600)  # Scale height back to original size
roi = frame[int(y):int(y+h), int(x):int(x+w)]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Calculate histogram for the initial ROI
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Set criteria for the CamShift algorithm
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Create a CSV file to store bounding box information
csv_file = open('camshift.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# Initialize track_window here
track_window = (int(x), int(y), int(w), int(h))  # Ensure they are integers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply Histogram Adjustment (Adaptive Histogram Equalization)
    # Perform CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    # Apply CamShift for the single ROI with updated histogram
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Convert track_window to tuple format
    track_window = tuple(map(int, track_window))

    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # Get the bounding box coordinates
    x, y, w, h = track_window
    x1, y1 = x, y
    bbox = [x1, y1, w, h]
    csv_writer.writerow([x1, y1, w, h])

    # Kalman Filter prediction
    prediction = kalman.predict()
    kalman.correct(np.array([[x1], [y1]], dtype=np.float32))

    # Update the tracking window based on Kalman Filter prediction
    track_window = (prediction[0], prediction[1], w, h)

    # Draw the tracking window for the object
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    # Write the frame with the tracking window to the output video
    out.write(frame)

    # Display the frame with the tracking window
    cv2.imshow('Single Object Tracking', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture, close CSV file, and windows
csv_file.close()
cap.release()
out.release()
cv2.destroyAllWindows()
