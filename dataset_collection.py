import cv2
import os

# -----------------------------
# Get person's name
# -----------------------------
name = input("Enter person's name: ").strip()

if name == "":
    print("Name cannot be empty.")
    exit()

# Create folder
folder = os.path.join("dataset", name)
os.makedirs(folder, exist_ok=True)

# Load Haar Cascade
face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera.")
    exit()

count = 0

print("\nStarting face capture...")
print("Look at the camera.")
print("Press 'q' to stop.\n")

while True:

    ret, frame = camera.read()

    if not ret:
        print("Unable to read camera.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        count += 1

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Save face
        face_image = gray[y:y + h, x:x + w]

        filename = os.path.join(
            folder,
            f"{count}.jpg"
        )

        cv2.imwrite(filename, face_image)

        cv2.putText(
            frame,
            f"Images: {count}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Capture Faces", frame)

    # Stop after 50 images
    if count >= 50:
        print("50 face images captured.")
        break

    # Press q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Face images saved in:", folder)
