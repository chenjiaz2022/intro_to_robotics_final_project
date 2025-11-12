import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np

class FinalProject(Node):
    def __init__(self):
        super().__init__('final_project')

def main(args=None):
    rclpy.init(args=args)
    node = StopAtWall()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()