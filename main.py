import cv2
import numpy as np
from iou import iou
from keras.models import load_model

cam = cv2.VideoCapture(1)

model = load_model("./models/ball_bbox_classifier",
                   custom_objects={"iou": iou})

cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Ball_bbox", cv2.WINDOW_KEEPRATIO)

width = 150
height = 200

while True:
    _, image = cam.read()

    cropped = image[:480, :360]

    resized = cv2.resize(cropped, (width, height),
                         interpolation=cv2.INTER_AREA)

    pred = model.predict(resized.reshape(1, 200, 150, 3))

    bbox = np.concatenate((
        np.round(pred[1][0][:2] * height), 
        np.round(pred[1][0][2:] * width)
    )).astype("uint8")


    if pred[0][0] > 0.5:
        cv2.rectangle(resized, (bbox[2], bbox[0]),
                      (bbox[3], bbox[1]), (255, 0, 0), 3)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    cv2.imshow("Camera", image)
    cv2.imshow("Ball_bbox", resized)

cam.release()
cv2.destroyAllWindows()
