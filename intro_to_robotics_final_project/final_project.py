import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
import numpy as np
import os
import time
from omx_cpp_interface.msg import ArmJointAngles, ArmGripperPosition

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

        # Set up subscribers
        self.scan_topic = f'/tb{ros_domain_id}/scan'
        self.laser_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.image_topic = f'/tb{ros_domain_id}/oakd/rgb/preview/image_raw/compressed'
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        
        # Useful attributes
        self.front_dist = 10
        self.next_task = None

        # Initialize Arm to initial position
        joint_msg = ArmJointAngles(joint1=0.0, joint2=0.0, joint3=0.0, joint4=0.0)
        self.arm_pub.publish(joint_msg)
        time.sleep(2)
        self.get_logger().info(f'Arm Initialized')

        self.task1_arm()

    def scan_callback(self, msg):
        "Function used to detect distance between the object and the turtlebot"
        mid = len(msg.ranges) // 4
        min_front_dist = np.min(msg.ranges[mid - 10 : mid + 10])
        self.front_dist = min_front_dist
        #self.get_logger().info(f'Distance is: {min_front_dist}')

    def image_callback(self, msg):
        "Function used to detect the colored blocks"
        
        # Note: It follows Sarah's advice and uses compressed images rather than raw images
        image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        pass

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

def main(args=None):
    rclpy.init(args=args)
    node = FinalProject()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()