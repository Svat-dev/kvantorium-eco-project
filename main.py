from datetime import datetime

import cv2
from ultralytics import YOLO

from constants import classes
from json_utils import add_overlap_object, restart_json
from utils import get_the_biggest, rectangle_overlap_percentage

restart_json("output.json")

overlap_threshold = input("Enter overlap threshold (default: 0.5): ")
if overlap_threshold.isdigit():
    overlap_threshold = float(overlap_threshold)
else:
    print("Invalid input for overlap threshold. Using default value of 0.5.")
    overlap_threshold = 0.5

# Load the YOLO11 model
boat_model = YOLO("runs/detect/kaka2/weights/best.pt")
human_model = YOLO("runs/detect/train/weights/best.pt")

# Open the video file
# video_path = "./assets/videos/walking-human-1.mp4"
video_path = "./assets/videos/human-and-boat.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        break

    current_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Run YOLO11 tracking on the frame, persisting tracks between frames
    boat_box = boat_model.predict(frame)
    human_box = human_model.predict(frame)

    boat_xy = boat_box[0].boxes.xyxy
    human_xy = human_box[0].boxes.xyxy

    if len(boat_xy) == 0:
        print("No boat detected")
        boat_xy = [[0, 0, 0, 0]]

    if len(human_xy) == 0:
        print("No human detected")
        human_xy = [[0, 0, 0, 0]]

    for i in boat_xy:
        cv2.rectangle(
            frame,
            (int(i[0]), int(i[1])),
            (int(i[2]), int(i[3])),
            (0, 255, 0),
            2,
        )

    for i in human_xy:
        cv2.rectangle(
            frame,
            (int(i[0]), int(i[1])),
            (int(i[2]), int(i[3])),
            (255, 0, 0),
            2,
        )

    for boat in boat_xy:
        for human in human_xy:
            overlap_percentage = rectangle_overlap_percentage(boat, human)
            if overlap_percentage > overlap_threshold:
                with open("output.txt", "a") as file:
                    file.write(f"{overlap_percentage * 100:.2f}%\n")
                add_overlap_object(
                    {
                        "track_id": current_frame_id,
                        "class": {"id": 2, "name": classes[2]},
                        "scores": float(f"{overlap_percentage:.3f}"),
                        "bbox": [620, 370, 710, 460],
                        "mask": [[[620, 370], [710, 370], [710, 460], [620, 460]]],
                        "timestamp": datetime.now().isoformat(),
                    },
                    "output.json",
                )

    cv2.imshow("Combined Predictions", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
get_the_biggest()
