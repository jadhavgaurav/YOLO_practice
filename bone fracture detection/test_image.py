from ultralytics import YOLO

# Load model
model = YOLO('runs/detect/train/weights/best.pt')  # update path if needed

test_img = 'test_xray.jpg'  # path to your test image

# Predict on image
results = model(test_img, show=True)  # show=True to display the result

# Optional: save results
results[0].save(filename='test_result.jpg')

