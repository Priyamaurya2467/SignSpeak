import cv2
import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Get the project path reliably
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"


def main():

    # Check model exists
    if not MODEL_PATH.exists():
        print("❌ Hand landmarker model not found!")
        print(f"Expected location: {MODEL_PATH}")
        return

    print(f"✅ Model found: {MODEL_PATH}")

    # MediaPipe options
    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Create hand landmarker
    detector = vision.HandLandmarker.create_from_options(options)

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open Camera")
        detector.close()
        return

    print("✅ Camera started")
    print("Press Q to quit")

    while True:

        success, frame = cap.read()

        if not success:
            print("❌ Failed to read frame")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hands
        result = detector.detect(mp_image)

        # Hand connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        # Draw detected hands
        for hand_landmarks in result.hand_landmarks:


            print("Number of landmarks:", len(hand_landmarks))

            for i,landmark in enumerate(hand_landmarks):
                print(
                    i,
                    "x:", round(landmark.x,4),
                    "y:", round(landmark.y,4),
                    "z:", round(landmark.z,4)
                )
                break

            # Draw points
            for landmark in hand_landmarks:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw connections
            for start, end in connections:

                x1 = int(
                    hand_landmarks[start].x *
                    frame.shape[1]
                )

                y1 = int(
                    hand_landmarks[start].y *
                    frame.shape[0]
                )

                x2 = int(
                    hand_landmarks[end].x *
                    frame.shape[1]
                )

                y2 = int(
                    hand_landmarks[end].y *
                    frame.shape[0]
                )

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

        # Display
        cv2.imshow(
            "SignSpeak - Hand Detection",
            frame
        )

        # Quit with Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()