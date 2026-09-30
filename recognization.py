import cv2
import os

# -----------------------------
# Check LBPH
# -----------------------------
if not hasattr(cv2, "face"):
    print("cv2.face is not available.")
    print("Install opencv-contrib-python")
    exit()

# -----------------------------
# Load LBPH recognizer
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(
    "trainer/trainer.yml"
)

# -----------------------------
# Load face detector
# -----------------------------
face_detector = cv2.CascadeClassifier(
    
    "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Load names
# -----------------------------
names = {}

if os.path.exists("trainer/names.txt"):

    with open("trainer/names.txt", "r") as file:

        for line in file:

            line = line.strip()

            if line:

                person_id, person_name = line.split(",", 1)

                names[int(person_id)] = person_name

else:

    print("names.txt not found.")
    exit()

# -----------------------------
# Open webcam
# -----------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Could not open camera.")
    exit()

print("Face recognition started.")
print("Press 'q' to exit.")

while True:

    ret, frame = camera.read()

    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # Process each face
    for (x, y, w, h) in faces:

        # Extract face
        face = gray[y:y + h, x:x + w]

        # Recognize face
        person_id, confidence = recognizer.predict(face)

        # Convert LBPH confidence to approximate percentage
        recognition_score = max(
            0,
            min(
                100,
                100 - confidence
            )
        )

        # Get name
        if person_id in names:

            name = names[person_id]

        else:

            name = "Unknown"

        # If confidence is too low, mark Unknown
        if confidence > 80:

            name = "Unknown"

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display name
        cv2.putText(
            frame,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # Display confidence
        cv2.putText(
            frame,
            f"Score: {recognition_score:.1f}%",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    # Display window
    cv2.imshow(
        "Face Detection and Recognition",
        frame
    )

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
