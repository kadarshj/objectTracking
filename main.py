# Using defaultdict so we dont have to
# do if key in dict checks
from collections import defaultdict
# For distance calculations
import math
# For opening, reading, and writing video frames
import cv2
# For array operations
import numpy as np
# Our custom detector
from detector import YoloV8ImageObjectDetection

def main():
    to_track = input("What would you like to track? ").strip().lower()

    # The YOLOv8 Detection Wrapper We Will Use
    # To Analyze Frames
    detector = YoloV8ImageObjectDetection()

    if (not detector.is_detectable(to_track)):
        raise ValueError(f"Error: My detecto does not know how to detect {to_track}!")

    # Create a video capture instance.
    # VideoCapture(0) corresponds to your computers
    # webcam
    # Open the video file
    video_path = "video_1.mp4"
    cap = cv2.VideoCapture(video_path)
    #cap = cv2.VideoCapture(0)


    # Lets grab the frames-per-second (FPS) of the
    # webcam so our output has a similar FPS.
    # Lets also grab the height and width so our
    # output is the same size as the webcam
    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_width  = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Now lets create the video writer. We will
    # write our processed frames to this object
    # to create the processed video.
    out = cv2.VideoWriter('outpy.avi', 
        cv2.VideoWriter_fourcc('M','J','P','G'), 
        fps, 
        (frame_width,frame_height)
    )

    cv2.namedWindow('Video')
    
    # The object tracks we have seen before
    track_history = defaultdict(lambda: [])

    # Previous frames, a frame count, and distance/speed
    # variables
    prev = None
    count = 1
    dist = 0
    speed = 0

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            continue

        # Use our detector to plot the bounding boxes on the frame,
        # give us our bounding boxes, and our object tracks
        frame, boxes, track_ids = detector.detect(frame, to_track)

        # For each bounding box and track we found,
        # We can calculate the box center and draw it and
        # the track on the screen. Tracks will be represented
        # as polylines created from our track
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 60: # Only hold the most recent 60 tracks
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Add the distance between the previous box center and this box center
            # to help us keep track of the total pixel distance
            if prev:
                dist += math.hypot(float(x)-float(prev[0]), float(y)-float(prev[1]))

            # Update our previous pointer
            prev = (float(x), float(y))

        # Calculate speed as total pixel distance / number of frames so that
        # we get average pixels moved / frame
        speed = dist / count
        count += 1
        cv2.putText(frame, f"Distance Covered (pixels): {dist:.5f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(frame, f"Average Speed (pixes/frame): {speed:.5f} ", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)

        # Write to our output file
        out.write(frame)

        # Show the frame
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()