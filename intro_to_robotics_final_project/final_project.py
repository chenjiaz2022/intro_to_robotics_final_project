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
from .ik_solver import OpenManipulatorIK

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
        self.ik_solver = OpenManipulatorIK()

        # Wait for publishers to initialize
        time.sleep(3)

        # Bridge for compressed images
        self.bridge = cv_bridge.CvBridge()
        cv2.namedWindow("window", 1)

        # Set up subscribers. topic first, then callback
        self.scan_topic = f'/tb{ros_domain_id}/scan'
        self.laser_sub = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, 10
        )

        self.image_topic = f'/tb{ros_domain_id}/oakd/rgb/preview/image_raw/compressed'
        self.image_sub = self.create_subscription(
            CompressedImage, self.image_topic, self.image_callback, 10
        )

        # Gesture subscriber (from gesture.py)
        self.gesture_topic = f'/tb{ros_domain_id}/hand_movement'
        self.gesture_sub = self.create_subscription(
            String, self.gesture_topic, self.gesture_callback, 10
        )
        self.get_logger().info(f'Subscribed to gesture topic {self.gesture_topic}')

        # Useful attributes
        self.front_dist = 10.0
        self.left_dist = 10.0
        self.right_dist = 10.0
        self.next_task = None
        self.cy_tag = 0

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
        # For gesture debouncing: require same gesture for 3 seconds
        self.last_gesture_raw = None
        self.last_gesture_time = None
        self.priority_over_avoidance = False
        self.start_sequence = False

        # 0 means idle, 1 means looking for AR tags and approaching
        self.next_goal = 0

        # 1 means resting in the corner, 2 means dancing with the user
        # 3 means grabbing the bottle, 4 means putting down the bottle
        self.current_task = 0

        # Parameter for area-based stopping (fraction of image area)
        self.min_tag_area_percent = 0.05  # 5% of the image area

        # ----------------- NAVIGATION STATE (SQUARE + AVOID) -----------------
        # nav_mode:
        #   "square_forward", "square_spin", "square_turn90"  -> square search path
        #   "avoid_..."                                       -> rectangular obstacle detour
        self.nav_mode = "square_forward"
        self.avoid_side = 1          # +1 = left, -1 = right
        self.avoid_phase_start = 0.0

        # For initial kick-off straight when starting search
        self.search_start_time = None

        # Square-path bookkeeping
        self.square_side_idx = 0     # 0,1,2,3 → which edge of the square
        self.square_phase_start = None
        # ---------------------------------------------------------------------

        # Initialize Arm and Gripper to initial position
        joint_msg = ArmJointAngles(joint1=0.0, joint2=0.0, joint3=0.0, joint4=0.0)
        self.arm_pub.publish(joint_msg)
        time.sleep(2)
        self.get_logger().info(f'Arm Initialized')

        gripper_msg = ArmGripperPosition(left_gripper=0.010, right_gripper=0.010)
        self.gripper_pub.publish(gripper_msg)
        time.sleep(2)
        self.get_logger().info('Gripper Openned')

    def scan_callback(self, msg: LaserScan):
        """
        Use LiDAR to detect distances in front and slightly to left/right.
        This is used for obstacle avoidance when no AR tag is visible.
        """
        ranges = np.asarray(msg.ranges, dtype=float)

        # Replace invalid/too-small readings with a large number
        ranges[ranges < 0.2] = 100.0

        n = len(ranges)
        mid = n // 4

        # Helper to clamp indices
        def clamp_slice(start, end):
            start = max(0, start)
            end = min(n, end)
            if start >= end:
                return np.array([100.0])
            return ranges[start:end]

        # "Front" sector: narrow slice in the middle
        front_slice = clamp_slice(mid - 10, mid + 10)
        self.front_dist = float(np.min(front_slice))

        # Slightly off-center left and right sectors for choosing avoidance direction
        left_slice = clamp_slice(mid + 20, mid + 60)
        right_slice = clamp_slice(mid - 60, mid - 20)

        self.left_dist = float(np.min(left_slice))
        self.right_dist = float(np.min(right_slice))
        

    def _do_obstacle_avoidance(self, use_initial_straight: bool):
        """
        One step of the navigation / obstacle-avoidance state machine.

        Search behavior (tag NOT visible):
        - Follow a SQUARE path:
            square_forward → square_spin (360°) → square_turn90 (90° to next side)
        - If an obstacle appears on a side, run the original rectangular avoid_* detour,
            then resume the SAME side of the square (effectively lengthening that side).

        use_initial_straight:
        - True  -> drive a short initial straight segment before starting the square
        - False -> used when tag is visible but blocked; we still use the same
                    avoid_* logic, but skip the initial search kick-off.
        """
        twist = Twist()
        now = time.time()

        # Tunable parameters
        emergency_stop_dist = 0.22     # hard stop
        safe_front_dist     = 0.55     # obstacle "in the way" threshold

        forward_speed       = 0.12
        turn_speed          = 0.6

        # Square search timing (seconds) – tune these on the robot
        side_time           = 9.0      # base time to drive each edge
        turn90_time         = 2.7      # approx time for a 90° turn
        spin360_time        = 4 * turn90_time  # 360° = 4 * 90°

        # Short initial straight kick-off when starting the whole search
        initial_straight_time = 5.33
        
        # 0) Emergency stop if something is extremely close
        if self.front_dist < emergency_stop_dist:
            self.cmd_pub.publish(Twist())
            return

        # 1) An initial ~180° turn, not forward motion <<<
        if use_initial_straight and not self.nav_mode.startswith("avoid_"):
            if self.search_start_time is None:
                self.search_start_time = now
            if now - self.search_start_time < initial_straight_time:
                # First, always turn in place (~180°) before starting the square path
                twist.linear.x = 0.0
                twist.angular.z = turn_speed
                self.cmd_pub.publish(twist)
                return
        else:
            # For tag-visible avoidance, we don't use this initial phase
            self.search_start_time = None

        # 2) If we are currently in a rectangular detour, finish that first
        if self.nav_mode.startswith("avoid_"):
            # Original rectangle parameters
            phase_turn_time = 2.5
            phase_fwd_time  = 4.0 + 1.0

            if self.nav_mode == "avoid_turn1":
                # First corner: rotate away from obstacle
                if now - self.avoid_phase_start < phase_turn_time:
                    twist.linear.x = 0.0
                    twist.angular.z = turn_speed * self.avoid_side
                else:
                    self.nav_mode = "avoid_forward1"
                    self.avoid_phase_start = now
                    twist.linear.x = forward_speed
                    twist.angular.z = 0.0

            elif self.nav_mode == "avoid_forward1":
                # Move in that direction (first forward edge)
                if now - self.avoid_phase_start < phase_fwd_time:
                    twist.linear.x = forward_speed
                    twist.angular.z = 0.0
                else:
                    self.nav_mode = "avoid_turn2"
                    self.avoid_phase_start = now
                    twist.linear.x = 0.0
                    twist.angular.z = -turn_speed * self.avoid_side

            elif self.nav_mode == "avoid_turn2":
                # Turn back to be parallel to original line
                if now - self.avoid_phase_start < phase_turn_time:
                    twist.linear.x = 0.0
                    twist.angular.z = -turn_speed * self.avoid_side
                else:
                    self.nav_mode = "avoid_forward2"
                    self.avoid_phase_start = now
                    twist.linear.x = forward_speed
                    twist.angular.z = 0.0

            elif self.nav_mode == "avoid_forward2":
                # Move parallel to the original line (passing the obstacle)
                if now - self.avoid_phase_start < (phase_fwd_time + 0.5):
                    twist.linear.x = forward_speed
                    twist.angular.z = 0.0
                else:
                    # Now start coming back toward the original line
                    self.nav_mode = "avoid_turn3"
                    self.avoid_phase_start = now
                    twist.linear.x = 0.0
                    twist.angular.z = -turn_speed * self.avoid_side

            elif self.nav_mode == "avoid_turn3":
                # Turn to head back toward the original straight line
                if now - self.avoid_phase_start < phase_turn_time:
                    twist.linear.x = 0.0
                    twist.angular.z = -turn_speed * self.avoid_side
                else:
                    self.nav_mode = "avoid_forward3"
                    self.avoid_phase_start = now
                    twist.linear.x = forward_speed
                    twist.angular.z = 0.0

            elif self.nav_mode == "avoid_forward3":
                # Move sideways back toward the original straight line
                if now - self.avoid_phase_start < phase_fwd_time:
                    twist.linear.x = forward_speed
                    twist.angular.z = 0.0
                else:
                    # Final turn to restore original heading
                    self.nav_mode = "avoid_turn4"
                    self.avoid_phase_start = now
                    twist.linear.x = 0.0
                    twist.angular.z = turn_speed * self.avoid_side

            elif self.nav_mode == "avoid_turn4":
                # Turn back to original orientation, then resume square search
                if now - self.avoid_phase_start < phase_turn_time:
                    twist.linear.x = 0.0
                    twist.angular.z = turn_speed * self.avoid_side
                else:
                    # Finished the rectangle; back to following the SAME square side.
                    # Reset square side timing so this side effectively becomes longer.
                    self.nav_mode = "square_forward"
                    self.square_phase_start = None
                    twist.linear.x = forward_speed
                    twist.angular.z = 0.0

            else:
                # Fallback: jump back to square search
                self.nav_mode = "square_forward"
                self.square_phase_start = None
                twist.linear.x = forward_speed
                twist.angular.z = 0.0

            self.cmd_pub.publish(twist)
            return

        if not use_initial_straight:
            return

        # 3) Not in a detour → run the square search pattern

        # Ensure nav_mode is one of the square states
        if self.nav_mode not in ("square_forward", "square_spin", "square_turn90"):
            self.nav_mode = "square_forward"
            self.square_phase_start = None

        if self.square_phase_start is None:
            self.square_phase_start = now
        elapsed = now - self.square_phase_start

        # If we hit an obstacle while on a side, start the rectangular detour
        if self.front_dist < safe_front_dist and self.nav_mode == "square_forward":
            if self.left_dist > self.right_dist:
                self.avoid_side = 1   # turn left
            else:
                self.avoid_side = -1  # turn right
            self.nav_mode = "avoid_turn1"
            self.avoid_phase_start = now
            twist.linear.x = 0.0
            twist.angular.z = turn_speed * self.avoid_side
            self.cmd_pub.publish(twist)
            return

        # Square states
        if self.nav_mode == "square_forward":
            # Drive along the current edge
            if elapsed < side_time:
                twist.linear.x = forward_speed
                twist.angular.z = 0.0
            else:
                # Reached the "corner" → start 360° spin
                self.nav_mode = "square_spin"
                self.square_phase_start = now
                twist.linear.x = 0.0
                twist.angular.z = turn_speed

        elif self.nav_mode == "square_spin":
            # Full 360° scan in place at the corner.
            # image_callback is still running and can see the tag while we spin.
            if elapsed < spin360_time:
                twist.linear.x = 0.0
                twist.angular.z = turn_speed
            else:
                # Done 360°, now rotate extra 90° to face next side
                self.nav_mode = "square_turn90"
                self.square_phase_start = now
                twist.linear.x = 0.0
                twist.angular.z = turn_speed

        elif self.nav_mode == "square_turn90":
            # Finish the 90° turn to align with the next edge
            if elapsed < turn90_time:
                twist.linear.x = 0.0
                twist.angular.z = turn_speed
            else:
                # Advance to next side in the square
                self.square_side_idx = (self.square_side_idx + 1) % 4
                self.nav_mode = "square_forward"
                self.square_phase_start = now
                twist.linear.x = forward_speed
                twist.angular.z = 0.0

        else:
            # Fallback: reset to square_forward
            self.nav_mode = "square_forward"
            self.square_phase_start = now
            twist.linear.x = forward_speed
            twist.angular.z = 0.0

        # Optional debug:
        self.get_logger().info(f"[NAV] mode={self.nav_mode}, side={self.square_side_idx}, front={self.front_dist:.2f}")
        self.cmd_pub.publish(twist)

    def image_callback(self, msg: CompressedImage):
        """
        Use camera to detect AR tags and move toward the tag chosen by the gesture.
        Stopping condition is based on the proportion of the image occupied by the tag.

        When tag is not visible but we have a navigation goal, we do:
          - Straight-line search forward
          - Obstacle avoidance only for obstacles directly in front
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
                # If obstacle is still on the way, first move around it
                safe_front_dist = 0.35  # same threshold as in avoidance
                if self.front_dist <= safe_front_dist and not self.priority_over_avoidance:
                    # Do avoidance WITHOUT initial straight phase (tag is visible)
                    self._do_obstacle_avoidance(use_initial_straight=False)
                    return

                # No obstacle blocking -> track tag normally
                self.nav_mode = "search_straight"
                self.search_start_time = None

                # tags[tag_id] = ((cx, cy), area)
                (cx_tag, cy_tag), area = tags[self.target_tag]
                self.cy_tag = cy_tag

                # Small bias
                cx_tag -= 5

                # P-controller on horizontal error
                err_x = (cx_tag - cx_img)
                ang_z = float(np.clip(-self.k_ang * err_x, -self.max_ang, self.max_ang))

                # Forward speed (will be clamped or zeroed if "close enough")
                lin_x = self.k_lin * 0.30
                lin_x = float(np.clip(lin_x, 0.0, self.max_lin))

                # Area-based stopping condition
                total_pixels = float(h * w)
                min_tag_area = total_pixels * self.min_tag_area_percent
                area_percent = (area / total_pixels) * 100.0

                # Stop when the tag fills enough of the frame
                if area >= min_tag_area:
                    self.priority_over_avoidance = True
                    self.get_logger().info(
                        f"Tag {self.target_tag} reached (area={area:.1f}, "
                        f"{area_percent:.1f}% of image)."
                    )

                # Never drive forward if LiDAR says too close
                if self.front_dist < 0.35:
                    lin_x = 0.0
                    ang_z = 0.0
                    self.start_sequence = True

                # Publish motion command
                twist = Twist()
                twist.linear.x = lin_x
                twist.angular.z = ang_z
                self.cmd_pub.publish(twist)

                if self.start_sequence:
                    if self.pending_return_tag != -1:
                        # e.g., gesture "5": go to tag 2, then back to tag 1
                        self.target_tag = self.pending_return_tag
                        self.pending_return_tag = -1
                        self.next_goal = 1
                        self.nav_mode = "square_forward"
                        self.search_start_time = None
                        self.square_phase_start = None
                        self.square_side_idx = 0
                        self.get_logger().info('Starting return to tag 1 after reaching tag 2.')

                        if self.current_task == 1:
                            self.arm_resting()
                            self.current_task = 0
                            self.start_sequence = False
                        elif self.current_task == 2:
                            self.arm_dancing()
                            self.current_task = 0
                            self.start_sequence = False
                        elif self.current_task == 3:
                            self.arm_grabbing()
                            self.current_task = 4
                            self.start_sequence = False
                        else:
                            self.arm_releasing()
                            self.current_task = 0
                            self.start_sequence = False
                    
                    else:
                        # Done with this goal
                        self.next_goal = 0
                        self.target_tag = -1
                        self.pending_return_tag = -1
                        self.current_gesture = "none"
                        self.nav_mode = "square_forward"
                        self.search_start_time = None
                        self.square_phase_start = None
                        self.square_side_idx = 0

                        if self.current_task == 1:
                            self.arm_resting()
                            self.current_task = 0
                            self.start_sequence = False
                        elif self.current_task == 2:
                            self.arm_dancing()
                            self.current_task = 0
                            self.start_sequence = False
                        elif self.current_task == 3:
                            self.arm_grabbing()
                            self.current_task = 4
                            self.start_sequence = False
                        else:
                            self.arm_releasing()
                            self.current_task = 0
                            self.start_sequence = False

            else:
                # Tag not detected -> straight-line search + obstacle avoidance
                self._do_obstacle_avoidance(use_initial_straight=True)

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

        # If gesture is not one of the valid ones, reset and ignore
        if gesture not in ("0", "2", "5"):
            self.last_gesture_raw = None
            self.last_gesture_time = None
            return

        now = time.time()

        # If gesture changed, start timing from now
        if gesture != self.last_gesture_raw:
            self.last_gesture_raw = gesture
            self.last_gesture_time = now
            # Not stable yet, just start the timer
            return

        # Check how long it's been stable
        if self.last_gesture_time is None:
            # Shouldn't really happen, but be safe
            self.last_gesture_time = now
            return

        elapsed = now - self.last_gesture_time
        if elapsed < 3.0:
            # Need at least 3 seconds of the same gesture
            return

        # At this point, the same valid gesture has been observed for >= 3 seconds
        # Accept it and reset the timer so we don't immediately re-trigger
        self.last_gesture_raw = None
        self.last_gesture_time = None

        self.current_gesture = gesture
        self.get_logger().info(f"Recognized gesture {gesture} (stable for {elapsed:.1f} s)")

        if gesture == "0":
            self.target_tag = 3
            self.pending_return_tag = -1
            self.current_task = 1
        elif gesture == "2":
            self.target_tag = 1
            self.pending_return_tag = -1
            self.current_task = 2
        elif gesture == "5":
            self.target_tag = 2
            self.pending_return_tag = 1
            self.current_task = 3

        self.next_goal = 1
        # Reset nav_mode when starting a new goal
        self.nav_mode = "square_forward"
        self.search_start_time = None
        self.square_phase_start = None
        self.square_side_idx = 0

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
            area = float(cv2.contourArea(pts))  # area of the tag polygon
            if tid in (1, 2, 3):
                # Store center and area
                out[int(tid)] = ((cx, cy), area)

            cv2.circle(dbg, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(dbg, f"id:{tid}", (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return out, dbg


    def arm_resting(self):
        "Arm movement for task 1 (resting in the corner)"
        joint_msg = ArmJointAngles(joint1=0.025, joint2=-0.420, joint3=0.920, joint4=1.120)
        self.arm_pub.publish(joint_msg)
        self.get_logger().info(f'Resting Arm')
        time.sleep(2)


    def arm_dancing(self):
        "Arm movement for task 2 (dancing when the user seems happy)"
        count = 0
        self.get_logger().info(f'Start dancing')

        joint1, joint2, joint3, joint4 = self.IK(0.08, 0.00, 0.37)
        joint_msg = ArmJointAngles(joint1=joint1, joint2=joint2, joint3=joint3, joint4=joint4)
        self.arm_pub.publish(joint_msg)
        time.sleep(2)

        joint1, joint2, joint3, joint4 = self.IK(0.13, 0.00, 0.28)
        joint_msg = ArmJointAngles(joint1=joint1, joint2=joint2, joint3=joint3, joint4=joint4)
        self.arm_pub.publish(joint_msg)
        time.sleep(2)

        joint1, joint2, joint3, joint4 = self.IK(0.08, 0.00, 0.37)
        joint_msg = ArmJointAngles(joint1=joint1, joint2=joint2, joint3=joint3, joint4=joint4)
        self.arm_pub.publish(joint_msg)
        time.sleep(2)

        while count < 2:
            joint1, joint2, joint3, joint4 = self.IK(0.06, -0.06, 0.37)
            joint_msg = ArmJointAngles(joint1=joint1, joint2=joint2, joint3=joint3, joint4=joint4)
            self.arm_pub.publish(joint_msg)
            time.sleep(4)

            joint1, joint2, joint3, joint4 = self.IK(0.09, -0.09, 0.28)
            joint_msg = ArmJointAngles(joint1=joint1, joint2=joint2, joint3=joint3, joint4=joint4)
            self.arm_pub.publish(joint_msg)
            time.sleep(2)

            joint1, joint2, joint3, joint4 = self.IK(0.06, -0.06, 0.37)
            joint_msg = ArmJointAngles(joint1=joint1, joint2=joint2, joint3=joint3, joint4=joint4)
            self.arm_pub.publish(joint_msg)
            time.sleep(2)

            joint1, joint2, joint3, joint4 = self.IK(0.06, 0.06, 0.37)
            joint_msg = ArmJointAngles(joint1=joint1, joint2=joint2, joint3=joint3, joint4=joint4)
            self.arm_pub.publish(joint_msg)
            time.sleep(4)

            joint1, joint2, joint3, joint4 = self.IK(0.09, 0.09, 0.28)
            joint_msg = ArmJointAngles(joint1=joint1, joint2=joint2, joint3=joint3, joint4=joint4)
            self.arm_pub.publish(joint_msg)
            time.sleep(2)

            joint1, joint2, joint3, joint4 = self.IK(0.06, 0.06, 0.37)
            joint_msg = ArmJointAngles(joint1=joint1, joint2=joint2, joint3=joint3, joint4=joint4)
            self.arm_pub.publish(joint_msg)
            time.sleep(2)

            count += 1

        joint1, joint2, joint3, joint4 = self.IK(0.29, 0.00, 0.21)
        joint_msg = ArmJointAngles(joint1=joint1, joint2=joint2, joint3=joint3, joint4=joint4)
        self.arm_pub.publish(joint_msg)
        time.sleep(2)
        self.get_logger().info(f'Dancing Complete. Return to initial position')


    def arm_grabbing(self):
        "Arm movement for task 3 (grabbing the bottle)"
        # send x and z to IK to calculate joint angles
        if self.cy_tag <= 120:
            z = 0.20
        else:
            z = 0.02
        joint1, joint2, joint3, joint4 = self.IK(self.front_dist, 0.0, z)

        joint_msg = ArmJointAngles(joint1=joint1, joint2=joint2, joint3=joint3, joint4=joint4)
        self.arm_pub.publish(joint_msg)
        self.get_logger().info(f'Move towards the object')
        time.sleep(5)

        gripper_msg = ArmGripperPosition(left_gripper=-0.010, right_gripper=-0.010)
        self.gripper_pub.publish(gripper_msg)
        self.get_logger().info('Gripper Closed')
        time.sleep(2)

    def arm_releasing(self):
        "Arm movement for task 4 (releasing the bottle)"
        # Put arm to initial position and open gripper
        joint_msg = ArmJointAngles(joint1=0.0, joint2=0.0, joint3=0.0, joint4=0.0)
        self.arm_pub.publish(joint_msg)
        time.sleep(2)
        self.get_logger().info(f'Arm Initialized')

        gripper_msg = ArmGripperPosition(left_gripper=0.010, right_gripper=0.010)
        self.gripper_pub.publish(gripper_msg)
        time.sleep(2)
        self.get_logger().info('Gripper Openned')

    def IK(self, x, y, z):
        target = [x, y, z]

        if not self.ik_solver.is_reachable(target):
            self.get_logger().warn(f"Target {target} is not reachable, using neutral pose instead")
            return 0.0, 0.0, 0.0, 0.0

        # Call the IK solver
        joints, err = self.ik_solver.inverse_kinematics(target_pos=target)

        if joints is None:
            self.get_logger().warn(f"IK failed for target {target}, using neutral pose")
            return 0.0, 0.0, 0.0, 0.0

        self.get_logger().info(
            f"IK solution for {target}: "
            f"q1={joints[0]:.3f}, q2={joints[1]:.3f}, "
            f"q3={joints[2]:.3f}, q4={joints[3]:.3f}, err={err:.4f}"
        )

        return float(joints[0]), float(joints[1]), float(joints[2]), float(joints[3])

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
