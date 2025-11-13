
import cv2
import numpy as np
import os
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import face_recognition
import mediapipe as mp

# -----------------------------
# Initialize models
# -----------------------------
detector = MTCNN()
emotion_model = load_model(r'C:\Users\chuka\Downloads\model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

# -----------------------------
# Load known faces
# -----------------------------
def load_known_faces(folder="known_faces"):
    known_encodings, known_names = [], []
    if not os.path.exists(folder):
        print(f"⚠ Folder '{folder}' not found.")
        return known_names, known_encodings

    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            img = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(img)
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
    print(f"[Info] {len(known_names)} known faces loaded.")
    return known_names, known_encodings

known_names, known_encodings = load_known_faces("known_faces")

# -----------------------------
# Helper for Text Drawing
# -----------------------------
def draw_label(image, text, x, y, color=(255, 255, 255)):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

# -----------------------------
# Process Image
# -----------------------------
image_path = r"C:\Users\chuka\OneDrive\Documents\TSU\AI for Cybersecurity\congress_wide.webp"
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError(f" Image not found at {image_path}")

rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img_h, img_w, _ = frame.shape

faces = detector.detect_faces(rgb_frame)

if not faces:
    print("No faces detected.")
else:
    print(f" Detected {len(faces)} face(s):\n")

for i, face in enumerate(faces, start=1):
    x, y, w, h = face['box']
    x, y = abs(x), abs(y)
    x2, y2 = x + w, y + h

    # -----------------------------
    # Face Recognition
    # -----------------------------
    top, right, bottom, left = y, x + w, y + h, x
    face_locations = [(top, right, bottom, left)]
    encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    name = "Unknown"

    if len(encodings) > 0:
        face_encoding = encodings[0]
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

    # -----------------------------
    # Emotion Detection
    # -----------------------------
    roi_gray = cv2.cvtColor(frame[y:y2, x:x2], cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, (48, 48))
    roi = roi_gray.astype('float') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    prediction = emotion_model.predict(roi, verbose=0)[0]
    emotion_label = emotion_labels[prediction.argmax()]

    # -----------------------------
    # Head Pose Estimation
    # -----------------------------
    cropped_rgb = rgb_frame[y:y2, x:x2]
    results = face_mesh.process(cropped_rgb)
    direction = "Forward"
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_3d, face_2d = [], []
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    x_l, y_l = int(lm.x * w), int(lm.y * h)
                    face_2d.append([x_l, y_l])
                    face_3d.append([x_l, y_l, lm.z])
            if len(face_2d) == 6:
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                cam_matrix = np.array([[w, 0, w / 2],
                                       [0, w, h / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                if success:
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles, *_ = cv2.RQDecomp3x3(rmat)
                    x_angle, y_angle, z_angle = angles[0] * 360, angles[1] * 360, angles[2] * 360

                    if y_angle < -10:
                        direction = "Left"
                    elif y_angle > 10:
                        direction = "Right"
                    elif x_angle < -10:
                        direction = "Down"
                    elif x_angle > 10:
                        direction = "Up"
                    else:
                        direction = "Forward"

    # -----------------------------
    # Print Output (Console)
    # -----------------------------
    print(f" Face #{i}")
    print(f"   Name: {name}")
    print(f"   Emotion: {emotion_label}")
    print(f"   Direction: {direction}\n")

    # -----------------------------
    # Draw on Image (Annotation)
    # -----------------------------
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 255), 2)
    draw_label(frame, f"Name: {name}", x, y - 25, (255, 255, 255))
    draw_label(frame, f"Emotion: {emotion_label}", x, y - 10, (0, 255, 0))
    draw_label(frame, f"Direction: {direction}", x, y + h + 15, (0, 255, 255))

# -----------------------------
# Save Annotated Output
# -----------------------------
output_path = os.path.splitext(image_path)[0] + "_annotated.jpg"
cv2.imwrite(output_path, frame)
print(f" Annotated image saved to:\n{output_path}")

cv2.imshow("Integrated Face System | Name | Emotion | Direction", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

