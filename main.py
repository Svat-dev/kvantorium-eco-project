import cv2
from ultralytics import YOLO

from utils import get_the_biggest, rectangle_overlap_percentage

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

    if success:
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
                if overlap_percentage > 0.5:
                    with open("output.txt", "a") as file:
                        file.write(f"{overlap_percentage * 100:.2f}%\n")
                    print(
                        f"Boat and human overlap detected with {overlap_percentage * 100:.2f}% overlap"
                    )

        cv2.imshow("Combined Predictions", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
get_the_biggest()
