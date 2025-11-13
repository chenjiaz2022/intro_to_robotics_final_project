import cv2 as cv
import numpy as np
from pythonosc import udp_client
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles

client = udp_client.SimpleUDPClient("127.0.0.1", 3333)

# Load dataset
data = np.load("Gesture_dataset.npz")
X_train = data["X"]  # shape (N, 63)
y_train = data["y"]  # shape (N,)
model_trained = True

print(f"Loaded dataset: {X_train.shape[0]} samples")

# Initialize Hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Open camera
cam = cv.VideoCapture(1)

def landmarks_to_feature(hand_landmarks):
    lm = hand_landmarks.landmark
    xs = np.array([p.x for p in lm])
    ys = np.array([p.y for p in lm])
    zs = np.array([p.z for p in lm])

    x0, y0, z0 = xs[0], ys[0], zs[0]
    xs = xs - x0
    ys = ys - y0
    zs = zs - z0

    norms = np.sqrt(xs**2 + ys**2 + zs**2)
    max_norm = np.max(norms)
    if max_norm > 0:
        xs /= max_norm
        ys /= max_norm
        zs /= max_norm

    feature = np.concatenate([xs, ys, zs], axis=0)
    return feature.astype(np.float32)

def predict_gesture(feature, k=5):
    if not model_trained or X_train is None or y_train is None:
        return "none"

    n_samples = X_train.shape[0]
    if n_samples == 0:
        return "none"

    k = min(k, n_samples)

    diffs = X_train - feature
    dists = np.linalg.norm(diffs, axis=1)

    knn_idx = np.argsort(dists)[:k]
    knn_labels = y_train[knn_idx]

    values, counts = np.unique(knn_labels, return_counts=True)
    majority_label = values[np.argmax(counts)]
    return str(int(majority_label))

print("Real-time Hand Gesture Recognition")
print("Press 'q' to quit.")

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        print("Camera frame not available")
        continue

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    hands_detected = hands.process(frame_rgb)

    movements = {'left': "none", 'right': "none"}

    if hands_detected.multi_hand_landmarks:
        for hand_landmarks, hand_class in zip(
            hands_detected.multi_hand_landmarks,
            hands_detected.multi_handedness
        ):
            hand_label = hand_class.classification[0].label.lower()

            drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )

            feat = landmarks_to_feature(hand_landmarks)
            gesture = predict_gesture(feat)
            movements[hand_label] = gesture

    # Decide what to send
    detected_gestures = [g for g in movements.values() if g != "none"]
    if detected_gestures:
        gesture_message = detected_gestures[0]
    else:
        gesture_message = "none"

    client.send_message("/hand_movement", gesture_message)

    debug_message = f"left hand {movements['left']}, right hand {movements['right']}"
    cv.putText(frame, debug_message, (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.imshow("Real-time Gesture", frame)
    print(debug_message, "-> sent:", gesture_message)

    key = cv.waitKey(20) & 0xFF
    if key == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
