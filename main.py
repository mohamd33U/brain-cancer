import cv2
import math
from ultralytics import YOLO

# Load the image
img = cv2.imread('bb.jpeg')

# Load the YOLO model
model = YOLO('best (2).pt')

# Class name
class_name = ['cancer_brain']

# Perform object detection
res = model(img)

# Flag to check if a valid bounding box was found
valid_box_found = False

# Iterate over the results
for r in res:
    boxes = r.boxes
    for box in boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Calculate width and height
        w, h = x2 - x1, y2 - y1

        # Get confidence score and class ID
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

        # Check if the detected class is 'cancer_brain' and confidence is above threshold
        if class_name[cls] == 'cancer_brain' and conf > 0.3:
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Put label on the bounding box
            cv2.putText(img, 'brain_cancer', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Set flag to True as a valid box is found
            valid_box_found = True

# If no valid box found, put the label at the fixed position
if not valid_box_found:
    cv2.putText(img, 'no_brain_cancer', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the image with bounding boxes
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

