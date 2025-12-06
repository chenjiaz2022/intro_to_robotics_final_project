import os
import cv2 as cv
import numpy as np
from pythonosc import udp_client
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv_bridge

client = udp_client.SimpleUDPClient("127.0.0.1", 3333)

# Load dataset
data = np.load("gesture_dataset.npz")
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

def landmarks_to_feature(hand_landmarks):
    """Convert raw MediaPipe landmark data to a normalized 63-dim vector."""
    lm = hand_landmarks.landmark
    xs = np.array([p.x for p in lm])
    ys = np.array([p.y for p in lm])
    zs = np.array([p.z for p in lm])

    # Center landmarks around wrist
    x0, y0, z0 = xs[0], ys[0], zs[0]
    xs = xs - x0
    ys = ys - y0
    zs = zs - z0

    # Normalize by maximum spread of points
    norms = np.sqrt(xs**2 + ys**2 + zs**2)
    max_norm = np.max(norms)
    if max_norm > 0:
        xs /= max_norm
        ys /= max_norm
        zs /= max_norm

    feature = np.concatenate([xs, ys, zs], axis=0)
    return feature.astype(np.float32)

def predict_gesture(feature, k=5):
    """Perform K-nearest-neighbors classification on the hand feature vector."""
    if not model_trained or X_train is None or y_train is None:
        return "none"

    n_samples = X_train.shape[0]
    if n_samples == 0:
        return "none"

    k = min(k, n_samples)

    # Compute Euclidean distances to all training samples
    diffs = X_train - feature
    dists = np.linalg.norm(diffs, axis=1)

    # Choose nearest neighbors
    knn_idx = np.argsort(dists)[:k]
    knn_labels = y_train[knn_idx]

    # Return most frequent label
    values, counts = np.unique(knn_labels, return_counts=True)
    majority_label = values[np.argmax(counts)]
    return str(int(majority_label))

class GestureRecognizer(Node):
    def __init__(self):
        super().__init__('gesture_recognizer')

        # Get ROS_DOMAIN_ID (same style as in main.py)
        ros_domain_id = os.getenv("ROS_DOMAIN_ID", "0")
        try:
            if int(ros_domain_id) < 10:
                ros_domain_id = "0" + str(int(ros_domain_id))
            else:
                ros_domain_id = str(int(ros_domain_id))
        except Exception:
            ros_domain_id = "00"

        self.bridge = cv_bridge.CvBridge()

        # Subscribe to robot camera topic (robot camera, not computer webcam)
        self.image_topic = f'/tb{ros_domain_id}/oakd/rgb/preview/image_raw'
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )

        # Publish recognized gesture as ROS topic
        self.gesture_topic = f'/tb{ros_domain_id}/hand_movement'
        self.gesture_pub = self.create_publisher(String, self.gesture_topic, 10)

        cv.namedWindow("Real-time Gesture", 1)

        self.get_logger().info(
            f"Real-time Hand Gesture Recognition using {self.image_topic}. "
            f"Publishing gestures to {self.gesture_topic}. Press 'q' in the window to quit."
        )

    def image_callback(self, msg: Image):
        """Process camera frames and publish gesture predictions."""
        # Convert ROS image to OpenCV BGR frame
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        hands_detected = hands.process(frame_rgb)

        movements = {'left': "none", 'right': "none"}

        # If one or more hands detected, classify each hand
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

                # Convert to feature and classify
                feat = landmarks_to_feature(hand_landmarks)
                gesture = predict_gesture(feat)
                movements[hand_label] = gesture

        # Choose one gesture to broadcast
        detected_gestures = [g for g in movements.values() if g != "none"]
        if detected_gestures:
            gesture_message = detected_gestures[0]
        else:
            gesture_message = "none"

        # Publish via ROS
        msg_out = String()
        msg_out.data = gesture_message
        self.gesture_pub.publish(msg_out)

        # Send via OSC
        client.send_message("/hand_movement", gesture_message)

        debug_message = f"left hand {movements['left']}, right hand {movements['right']}"
        cv.putText(frame, debug_message, (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.imshow("Real-time Gesture", frame)
        print(debug_message, "-> sent:", gesture_message)

        # Quit window
        key = cv.waitKey(5) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Exiting gesture recognizer.")
            rclpy.shutdown()

def main(args=None):
    print("Real-time Hand Gesture Recognition (Robot Camera)")
    print("Press 'q' in the image window to quit.")
    rclpy.init(args=args)
    node = GestureRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
