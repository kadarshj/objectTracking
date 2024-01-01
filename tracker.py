from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "video_2.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])
# Store the Object ID to track
to_track = 0
# Store the count of frame
count = 0
# Store the Object ID to track from List
cTrack = 0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        print('track ids')
        print(track_ids)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        print("to_track", to_track)
        if to_track == 0:
            count = count + 1
            img = cv2.resize(annotated_frame, (600, 500))
            cv2.imshow("YOLOv8 Tracking", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if count == 2: # This can be changed according to the frame from where objects need to be tracked
                to_track = input("What would you like to track? ").strip().lower()
                print('what')
                print(to_track)
                print(results[0])
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            print(f"track id to_track {track_id, to_track}")
            print(f"count {count}")
            cTrack = int(track_id)
            if cTrack == int(to_track) and count >= 2:
                print(f"track id to_track {track_id, to_track}")
                print(f"count {count}")
                x, y, w, h = box
                print('box')
                print(box)
                print(track_id)
                print(f"x y w h , {x, y, w, h}")
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

            # Draw the tracking lines
            if cTrack == int(to_track) and count == 2:
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0,0,65535), thickness=50)

        # Display the annotated frame
        imS = cv2.resize(annotated_frame, (600, 500))
        cv2.imshow("YOLOv8 Tracking", imS)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()