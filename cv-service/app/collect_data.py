import cv2
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from landmark import LandmarkCollector

Base_Dir =Path(__file__).resolve().parent.parent
MODEL_PATH = Base_Dir/"models"/"hand_landmarker.task"

def main():
    label = input("Enter label eg(A)")
    if not label:
        print("Detection not done")
        return

    print("Collectiong data from label")

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)

    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5

    )

    detector = vision.HandLandmarker.create_four_options(options)

    cap=cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        detector.close()
        return

    print()
    print("Camera started")
    print()
    print("Prece space to to capture")
    print("press q to exit")
    print()

    sample_count = 0

    while True:
        success,frame = cap.read()

        if not success:
            print("Failed to load frame")
            break

        frame = cv2.flip(frame,1)

        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

          # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        result = detector.detect(mp_image)

        # Draw landmarks
        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Draw points
            for landmark in hand:

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

        # Display information
        cv2.putText(
            frame,
            f"Sign: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "SPACE = Capture | Q = Quit",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "SignSpeak - Data Collection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Capture sample
        if key == ord(" "):

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                collector.save_landmarks(
                    hand,
                    label
                )

                sample_count += 1

                print(
                    f"✅ Sample {sample_count} saved"
                )

            else:

                print(
                    "⚠️ No hand detected"
                )

        # Quit
        elif key == ord("q"):

            break

    cap.release()

    cv2.destroyAllWindows()

    detector.close()

    print()
    print(
        f"✅ Collection finished. "
        f"{sample_count} samples saved for {label}"
    )


if __name__ == "__main__":
    main()
