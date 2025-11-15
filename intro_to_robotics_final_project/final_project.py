import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import os
import time
from omx_cpp_interface.msg import ArmJointAngles, ArmGripperPosition
from std_msgs.msg import String

import cv2
import cv_bridge


class FinalProject(Node):
    def __init__(self):
        super().__init__('final_project')

        # Get ROS_DOMAIN_ID
        ros_domain_id = os.getenv("ROS_DOMAIN_ID", "0")
        try:
            if int(ros_domain_id) < 10:
                ros_domain_id = "0" + str(int(ros_domain_id))
            else:
                ros_domain_id = str(int(ros_domain_id))
        except Exception:
            ros_domain_id = "00"
        self.get_logger().info(f'ROS_DOMAIN_ID: {ros_domain_id}')

        # Set up publishers
        self.cmd_vel_topic = f'/tb{ros_domain_id}/cmd_vel'
        self.arm_topic = f'/tb{ros_domain_id}/target_joint_angles'
        self.gripper_topic = f'/tb{ros_domain_id}/target_gripper_position'
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.arm_pub = self.create_publisher(ArmJointAngles, self.arm_topic, 10)
        self.gripper_pub = self.create_publisher(ArmGripperPosition, self.gripper_topic, 10)

        # Wait for publishers to initialize
        time.sleep(3)

        # Bridge for compressed images
        self.bridge = cv_bridge.CvBridge()
        cv2.namedWindow("window", 1)

        # Set up subscribers
        self.scan_topic = f'/tb{ros_domain_id}/scan'
        self.laser_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.image_topic = f'/tb{ros_domain_id}/oakd/rgb/preview/image_raw/compressed'
        self.image_sub = self.create_subscription(
            CompressedImage, self.image_topic, self.image_callback, 10
        )

        # Gesture subscriber (from Gesture.py)
        self.gesture_topic = f'/tb{ros_domain_id}/hand_movement'
        self.gesture_sub = self.create_subscription(
            String, self.gesture_topic, self.gesture_callback, 10
        )
        self.get_logger().info(f'Subscribed to gesture topic {self.gesture_topic}')

        # Useful attributes
        self.front_dist = 10.0
        self.next_task = None

        # Controller params for AR tag approach
        self.k_ang = 0.0025
        self.k_lin = 0.22
        self.max_ang = 1.5
        self.max_lin = 0.35
        self.center_px = 20

        # ArUco dictionaries
        self.aruco_dict = "DICT_4X4_50"
        self.last_tag_print = {1: 0.0, 2: 0.0, 3: 0.0}
        self.print_interval = 0.8

        # State for gesture-based navigation
        self.current_gesture = "none"
        self.target_tag = -1
        self.pending_return_tag = -1
        # 0 means idle, 1 means looking for AR tags and approaching
        self.next_goal = 0

        # --- parameter for area-based stopping (fraction of image area) ---
        self.min_tag_area_percent = 0.14  # 14% of the image area as in previous project

        # Initialize Arm to initial position (optional) -> comment for now
        # joint_msg = ArmJointAngles(joint1=0.0, joint2=0.0, joint3=0.0, joint4=0.0)
        # self.arm_pub.publish(joint_msg)
        # time.sleep(2)
        # self.get_logger().info(f'Arm Initialized')

        # self.task1_arm()

    def scan_callback(self, msg):
        """
        Function used to detect distance between the object and the turtlebot.
        We keep this as an extra safety signal (e.g., could be used to clamp speed),
        but we no longer use it as the main stopping condition.
        """
        mid = len(msg.ranges) // 4
        current_ranges_array = np.asarray(msg.ranges)
        current_ranges_array[current_ranges_array < 0.2] = 100
        min_front_dist = np.min(current_ranges_array[mid - 10: mid + 10])
        self.front_dist = float(min_front_dist)

    def image_callback(self, msg):
        """
        Use camera to detect AR tags and move toward the tag chosen by the gesture.
        Stopping condition is now based on the proportion of the image occupied by the tag.
        """
        image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

        bgr = image
        h, w = bgr.shape[:2]
        cx_img = w // 2

        # Show debug image
        cv2.imshow("window", bgr)
        cv2.waitKey(1)

        now = time.time()

        # If we have a target tag from a gesture, look for it and move
        if self.next_goal == 1 and self.target_tag != -1:
            tags, dbg = self._detect_aruco(bgr)
            if dbg is not None:
                bgr = dbg

            # Log detections at intervals
            for tid in (1, 2, 3):
                if tid in tags and (now - self.last_tag_print[tid]) > self.print_interval:
                    self.get_logger().info(f"Detected tag {tid}")
                    self.last_tag_print[tid] = now

            # If target tag detected
            if self.target_tag in tags:
                # tags[tag_id] = ((cx, cy), area)
                (cx_tag, cy_tag), area = tags[self.target_tag]

                # Small bias if you want (as before)
                cx_tag -= 5

                # P-controller on horizontal error
                err_x = (cx_tag - cx_img)
                ang_z = float(np.clip(-self.k_ang * err_x, -self.max_ang, self.max_ang))

                # Forward speed (will be clamped or zeroed if "close enough")
                lin_x = self.k_lin * 0.30
                lin_x = float(np.clip(lin_x, 0.0, self.max_lin))

                start_sequence = False

                # --- NEW: area-based stopping condition, like in TAG phase before ---
                total_pixels = float(h * w)
                min_tag_area = total_pixels * self.min_tag_area_percent
                area_percent = (area / total_pixels) * 100.0

                # Stop when the tag fills enough of the frame
                if area >= min_tag_area:
                    lin_x = 0.0
                    ang_z = 0.0
                    start_sequence = True
                    self.get_logger().info(
                        f"Tag {self.target_tag} reached (area={area:.1f}, "
                        f"{area_percent:.1f}% of image)."
                    )

                # Optional extra safety: never drive forward if LiDAR says too close
                if self.front_dist < 0.25:
                    lin_x = 0.0

                # Publish motion command
                twist = Twist()
                twist.linear.x = lin_x
                twist.angular.z = ang_z
                self.cmd_pub.publish(twist)

                # Only advance/chain goals once we consider the tag "reached"
                if start_sequence:
                    if self.pending_return_tag != -1:
                        # e.g., gesture "5": go to tag 2, then back to tag 1
                        self.target_tag = self.pending_return_tag
                        self.pending_return_tag = -1
                        self.next_goal = 1
                        self.get_logger().info('Starting return to tag 1 after reaching tag 2.')
                    else:
                        # Done with this goal
                        self.next_goal = 0
                        self.target_tag = -1

            else:
                # Rotate to search for tag
                twist = Twist()
                twist.angular.z = 0.25
                twist.linear.x = 0.0
                self.cmd_pub.publish(twist)

    def gesture_callback(self, msg: String):
        """
        Receive gesture from Gesture.py and set the AR tag goal.

        Mapping:
          gesture "0" -> move to AR tag 3
          gesture "2" -> move to AR tag 1
          gesture "5" -> move to AR tag 2, then back to AR tag 1
        """
        # If currently executing a navigation goal, ignore new gestures
        if self.next_goal == 1:
            return

        gesture = msg.data.strip()
        if gesture not in ("0", "2", "5"):
            return

        self.current_gesture = gesture
        self.get_logger().info(f"Received gesture {gesture}")

        if gesture == "0":
            self.target_tag = 3
            self.pending_return_tag = -1
        elif gesture == "2":
            self.target_tag = 1
            self.pending_return_tag = -1
        elif gesture == "5":
            self.target_tag = 2
            self.pending_return_tag = 1

        self.next_goal = 1

    def _detect_aruco(self, bgr):
        """
        Detect ArUco markers in the image and return their centers + areas with a debug overlay.

        Returns:
          out: dict[tag_id] = ((cx, cy), area)
          dbg: image with markers drawn
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dbg = bgr.copy()
        out = {}

        def make_params():
            """Define relaxed detection parameters for robustness against small/blurred tags."""
            p = cv2.aruco.DetectorParameters()
            p.adaptiveThreshWinSizeMin = 3
            p.adaptiveThreshWinSizeMax = 53
            p.adaptiveThreshWinSizeStep = 10
            p.minMarkerPerimeterRate = 0.02
            p.maxMarkerPerimeterRate = 4.0
            p.polygonalApproxAccuracyRate = 0.05
            p.minCornerDistanceRate = 0.01
            p.minDistanceToBorder = 1
            p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            return p

        def detect_with_dict(dict_name):
            """Attempt detection using a specified ArUco dictionary."""
            ar_dict_id = getattr(cv2.aruco, dict_name)
            ar_dict = cv2.aruco.getPredefinedDictionary(ar_dict_id)
            params = make_params()
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, ar_dict, parameters=params)
            return corners, ids, dict_name

        corners, ids, used_dict = detect_with_dict(self.aruco_dict)
        if corners is not None:
            cv2.aruco.drawDetectedMarkers(dbg, corners, ids)
        else:
            return out, dbg

        if ids is None or len(ids) == 0:
            return out, dbg

        ids = ids.flatten()
        for i, tid in enumerate(ids):
            pts = corners[i][0] if len(corners[i].shape) == 3 else corners[i]
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            area = float(cv2.contourArea(pts))  # NEW: area of the tag polygon
            if tid in (1, 2, 3):
                # Store center and area
                out[int(tid)] = ((cx, cy), area)

            cv2.circle(dbg, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(dbg, f"id:{tid}", (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return out, dbg

    def task1_arm(self):
        "Arm movement for task 1 (dancing when the user seems happy)"
        count = 0
        self.get_logger().info(f'Start dancing')

        joint_msg = ArmJointAngles(joint1=0.0, joint2=-0.9, joint3=0.0, joint4=0.0)
        self.arm_pub.publish(joint_msg)
        time.sleep(2)

        joint_msg = ArmJointAngles(joint1=0.0, joint2=-0.9, joint3=0.0, joint4=0.8)
        self.arm_pub.publish(joint_msg)
        time.sleep(2)

        joint_msg = ArmJointAngles(joint1=0.0, joint2=-0.9, joint3=0.0, joint4=0.0)
        self.arm_pub.publish(joint_msg)
        time.sleep(2)

        while count < 2:
            joint_msg = ArmJointAngles(joint1=-0.8, joint2=-0.9, joint3=0.0, joint4=0.0)
            self.arm_pub.publish(joint_msg)
            time.sleep(4)

            joint_msg = ArmJointAngles(joint1=-0.8, joint2=-0.9, joint3=0.0, joint4=0.8)
            self.arm_pub.publish(joint_msg)
            time.sleep(2)

            joint_msg = ArmJointAngles(joint1=-0.8, joint2=-0.9, joint3=0.0, joint4=0.0)
            self.arm_pub.publish(joint_msg)
            time.sleep(2)

            joint_msg = ArmJointAngles(joint1=0.8, joint2=-0.9, joint3=0.0, joint4=0.0)
            self.arm_pub.publish(joint_msg)
            time.sleep(4)

            joint_msg = ArmJointAngles(joint1=0.8, joint2=-0.9, joint3=0.0, joint4=0.8)
            self.arm_pub.publish(joint_msg)
            time.sleep(2)

            joint_msg = ArmJointAngles(joint1=0.8, joint2=-0.9, joint3=0.0, joint4=0.0)
            self.arm_pub.publish(joint_msg)
            time.sleep(2)

            count += 1

        joint_msg = ArmJointAngles(joint1=0.0, joint2=0.0, joint3=0.0, joint4=0.0)
        self.arm_pub.publish(joint_msg)
        time.sleep(2)
        self.get_logger().info(f'Dancing Complete. Return to initial position')

    def _stop(self):
        """Stop the robot by publishing zero velocity."""
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = FinalProject()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
