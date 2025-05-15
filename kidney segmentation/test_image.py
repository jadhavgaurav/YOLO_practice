from ultralytics import YOLO
import cv2
import numpy as np

# Load model
model = YOLO('runs/detect/train4/weights/best.pt')  # update path if needed

test_img = 'Normal- (86).jpg'  # path to your test image

# Predict on image
results = model(test_img, show=True)  # show=True to display the result

# Optional: save results
results[0].save(filename='test_result.jpg')

# img = cv2.imread('test_result.jpg')

# cv2.imshow('Test Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()