from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

model = YOLO('best.pt')
video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video file.")
else :
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      print("End of video.")
      break
    
    results = model(frame)[0]
    for result in results.boxes:
        x_min, y_min, x_max, y_max = map(int, result.xyxy[0].tolist())
        confidence = float(result.conf[0])
        class_id = int(result.cls[0])
        class_name = model.names[class_id]

        if class_name == "enem" or class_name == "enem_tank":
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        else :
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Inference Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  cap.release()
  cv2.destroyAllWindows()