import cv2
import os
import numpy as np

# -----------------------------
# Check LBPH availability
# -----------------------------
if not hasattr(cv2, "face"):
    print("cv2.face is not available.")
    print("Install opencv-contrib-python")
    exit()

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load Haar Cascade
detector = cv2.CascadeClassifier(
 
    "haarcascade_frontalface_default.xml"
)

faces = []
ids = []

# Dictionary for person names
names = {}

# Dataset directory
dataset_path = "dataset"

# Check dataset
if not os.path.exists(dataset_path):
    print("Dataset folder does not exist.")
    exit()

# Person ID
person_id = 0

# Read each person's folder
for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(
        dataset_path,
        person_name
    )

    if not os.path.isdir(person_folder):
        continue

    person_id += 1

    names[person_id] = person_name

    print(
        f"Processing: {person_name} "
        f"(ID = {person_id})"
    )

    # Read images
    for image_name in os.listdir(person_folder):

        image_path = os.path.join(
            person_folder,
            image_name
        )

        # Read image as grayscale
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        # Detect face
        detected_faces = detector.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (x, y, w, h) in detected_faces:

            face = image[y:y + h, x:x + w]

            faces.append(face)
            ids.append(person_id)

# Check data
if len(faces) == 0:
    print("No faces found in dataset.")
    exit()

# Convert IDs to NumPy array
ids = np.array(ids)

# Train recognizer
print("\nTraining model...")

recognizer.train(
    faces,
    ids
)

# Create trainer folder
os.makedirs("trainer", exist_ok=True)

# Save model
recognizer.write(
    "trainer/trainer.yml"
)

# Save names
with open("trainer/names.txt", "w") as file:

    for person_id, person_name in names.items():

        file.write(
            f"{person_id},{person_name}\n"
        )

print("\nTraining completed.")
print("Model saved as: trainer/trainer.yml")
print("Names saved as: trainer/names.txt")
