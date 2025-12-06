import os
import cv2 as cv
import numpy as np
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv_bridge

# Initialize the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Training storage
training_data = []   # list of feature vectors
training_labels = [] # list of ints (0,1,2,3)

current_features = {"left": None, "right": None}

def landmarks_to_feature(hand_landmarks):
    """
    Convert MediaPipe hand landmarks to a normalized feature vector.
    21 points -> 63 values (x,y,z normalized).
    """
    lm = hand_landmarks.landmark
    xs = np.array([p.x for p in lm])
    ys = np.array([p.y for p in lm])
    zs = np.array([p.z for p in lm])

    # Use wrist as origin
    x0, y0, z0 = xs[0], ys[0], zs[0]
    xs = xs - x0
    ys = ys - y0
    zs = zs - z0

    # Normalize by max distance from wrist
    norms = np.sqrt(xs**2 + ys**2 + zs**2)
    max_norm = np.max(norms)
    if max_norm > 0:
        xs /= max_norm
        ys /= max_norm
        zs /= max_norm

    feature = np.concatenate([xs, ys, zs], axis=0)
    return feature.astype(np.float32)

class GestureDatasetCollector(Node):
    def __init__(self):
        super().__init__('gesture_dataset_collector')

        # Get ROS_DOMAIN_ID
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

        cv.namedWindow("Dataset Collector", 1)

        self.get_logger().info(
            f"Hand Gesture Dataset Collector using {self.image_topic}. "
            "Show gesture 0/2/5 and press that key to record."
        )

    def image_callback(self, msg: Image):
        """Process each incoming camera frame, detect hands, and record samples."""
        global current_features, training_data, training_labels

        # Convert ROS2 image to OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Convert to RGB for MediaPipe
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        hands_detected = hands.process(frame_rgb)

        # Reset current frame’s feature vectors
        current_features["left"] = None
        current_features["right"] = None

        # Loop through detected hands and extract features
        if hands_detected.multi_hand_landmarks:
            for hand_landmarks, hand_class in zip(
                hands_detected.multi_hand_landmarks,
                hands_detected.multi_handedness
            ):
                hand_label = hand_class.classification[0].label.lower() # 'left' or 'right'

                # Draw skeleton on visualization frame
                drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )
                
                # Convert landmarks to feature vector
                feat = landmarks_to_feature(hand_landmarks)
                current_features[hand_label] = feat

        # Display UI instructions
        cv.putText(frame, "Press 0/2/5 to add sample, 's' to save, 'q' to quit.",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.imshow("Dataset Collector", frame)

        # Handle key presses
        key = cv.waitKey(20) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quitting dataset collection.")
            rclpy.shutdown()
            return

        # If recording a gesture
        if key in [ord('0'), ord('2'), ord('5')]:
            label = int(chr(key))
            saved_any = False
             # Save features from any detected hand
            for side in ["left", "right"]:
                feat = current_features[side]
                if feat is not None:
                    training_data.append(feat)
                    training_labels.append(label)
                    saved_any = True
            if saved_any:
                print(f"Added sample(s) for label {label}. Total: {len(training_labels)}")
            else:
                print("No hand detected when you pressed the key, nothing saved.")

        # Save dataset to disk
        if key == ord('s'):
            if len(training_labels) == 0:
                print("No samples to save yet.")
            else:
                X = np.vstack(training_data)
                y = np.array(training_labels, dtype=np.int32)
                np.savez("gesture_dataset.npz", X=X, y=y)
                print(f"Saved dataset with {len(y)} samples to gesture_dataset.npz")

print("Hand Gesture Dataset Collector")
print("Show gesture 0/2/5 to the camera, then press that number key to record a sample.")
print("Keys:")
print("  0/2/5 -> add training sample with that label")
print("  s       -> save dataset to gesture_dataset.npz")
print("  q       -> quit")

def main(args=None):
    rclpy.init(args=args)
    node = GestureDatasetCollector()
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
