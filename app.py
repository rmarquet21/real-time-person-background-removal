import os
import random
import cv2
import time
import numpy as np
from ultralytics import YOLO

list_model = [
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt",
]


# add choice for model selection 
print("Choose a model: ")
for i, model in enumerate(list_model):
    print(f"{i}: {model}")
model_choice = int(input("Enter the model number: "))
is_onnx = bool(int(input("Do you want to use ONNX? (0/1): ")))

model = YOLO(list_model[model_choice])

yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
colors = [random.choices(range(256), k=3) for _ in classes_ids]

if is_onnx and not os.path.exists(list_model[model_choice][:-3] + ".onnx"):
    model = model.export(format="onnx")
    model = YOLO(list_model[model_choice][:-3] + ".onnx")

# define a video capture object 
vid = cv2.VideoCapture(0) 
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if model_choice == 0 or model_choice == 1:
    person_class_id = 0
else:
    person_class_id = yolo_classes.index("person")  # Get the class ID for "person"

conf = 0.5

start_time = time.time()
frame_id = 0

while(True): 
    frame_id += 1
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    if not ret:
        break

    results = model.predict(frame, stream=True, conf=conf)
    
    # Filter masks for "person" class and remove background
    for result in results:
        for mask, box in zip(result.masks.xy, result.boxes):
            if int(box.cls[0]) == person_class_id:  # If the class is "person"
                points = np.int32([mask])
                black_background = np.zeros(frame.shape, dtype=np.uint8)
                cv2.fillPoly(black_background, points, (255, 255, 255))

    # Calculate and display FPS
    elapsed_time = time.time() - start_time
    fps = frame_id / elapsed_time

    frame = cv2.bitwise_and(frame, black_background)
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "FPS : " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv2.putText(frame, "Q to quit", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
