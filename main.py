from datetime import datetime

import cv2
from ultralytics import YOLO

from modules.constants import classes
from modules.json_utils import add_overlap_object, restart_json, restart_txt
from modules.utils import get_the_biggest, rectangle_overlap_percentage

restart_json("output.json")
restart_txt("output.txt")

overlap_threshold = 0.5
probs_threshold = 0.5

with open("params.txt", "r", encoding="utf-8") as file:
    params = file.read().splitlines()
    overlap_threshold = float(params[0].split(" = ")[1])
    probs_threshold = float(params[1].split(" = ")[1])

# Load the YOLO8 model
boat_model = YOLO("runs/detect/boat/weights/best.pt")
human_model = YOLO("runs/detect/human/weights/best.pt")

# Open the video file
# video_path = "./assets/images/man-and-boat-1.jpg"
video_path = "./assets/videos/human-and-boat.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        break

    current_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Run YOLO8 tracking on the frame, persisting tracks between frames
    boat_box = boat_model.predict(frame)
    human_box = human_model.predict(frame)

    boat_probs_array = boat_box[0].boxes.conf
    human_probs_array = human_box[0].boxes.conf

    boat_probs = (
        round(float(boat_probs_array[0]), 3) if len(boat_probs_array) > 0 else 0
    )
    human_probs = (
        round(float(human_probs_array[0]), 3) if len(human_probs_array) > 0 else 0
    )

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
            avg_probability = round((boat_probs + human_probs) / 2, 3)

            if (
                overlap_percentage >= overlap_threshold
                and avg_probability >= probs_threshold
            ):
                with open("output.txt", "a", encoding="utf-8") as file:
                    file.write(f"{overlap_percentage * 100:.2f}%\n")

                bbox = [
                    int(max(boat[0], human[0])),
                    int(max(boat[1], human[1])),
                    int(min(boat[2], human[2])),
                    int(min(boat[3], human[3])),
                ]
                mask = [
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                ]

                add_overlap_object(
                    {
                        "track_id": current_frame_id,
                        "class": {"id": 2, "name": classes[2]},
                        "scores": float(f"{overlap_percentage:.3f}"),
                        "probability": avg_probability,
                        "bbox": bbox,
                        "mask": mask,
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
