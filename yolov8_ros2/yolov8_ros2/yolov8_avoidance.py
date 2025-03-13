import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YOLOv8DynamicObjectAvoidanceNode(Node):
    def __init__(self):
        super().__init__('yolov8_dynamic_object_avoidance_node')
        self.image_pub = self.create_publisher(Image, 'yolov8/image', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.model = YOLO('/home/kabilankb/ultralytics/models/yolov8n.pt')
        self.br = CvBridge()

        # Subscriber to the camera topic
        self.image_sub = self.create_subscription(
            Image,
            '/jetson_webcam',  # Update this with your camera topic
            self.camera_callback,
            10
        )

    def camera_callback(self, msg):
        # Convert the ROS Image message to an OpenCV image
        cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Perform object detection
        results = self.model(cv_image)

        # Initialize the velocity message
        cmd_vel_msg = Twist()

        # Variables to determine the closest object in the robot's path
        closest_object_distance = float('inf')
        closest_object_center_x = None

        # Dynamic object avoidance logic
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

            for box, class_id in zip(boxes, class_ids):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.model.names[int(class_id)]  # Get the class name

                # Calculate the distance of the object from the robot
                object_distance = (y2 - y1)  # Assuming larger boxes are closer
                object_center_x = (x1 + x2) // 2

                # Update the closest object information if necessary
                if object_distance < closest_object_distance:
                    closest_object_distance = object_distance
                    closest_object_center_x = object_center_x

                # Draw the bounding box and label on the image
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Object avoidance decision based on the closest object
        if closest_object_center_x is not None:
            image_center_x = cv_image.shape[1] // 2

            if abs(closest_object_center_x - image_center_x) < cv_image.shape[1] * 0.2:
                # If the object is in the center region, adjust velocity
                cmd_vel_msg.linear.x = 0.0  # Stop the robot
                cmd_vel_msg.angular.z = 0.5  # Rotate to avoid the object
            else:
                # Otherwise, move forward
                cmd_vel_msg.linear.x = 0.2
                cmd_vel_msg.angular.z = 0.0
        else:
            # No objects detected, move forward
            cmd_vel_msg.linear.x = 0.2
            cmd_vel_msg.angular.z = 0.0

        # Publish the velocity command
        self.cmd_vel_pub.publish(cmd_vel_msg)

        # Convert the OpenCV image back to a ROS Image message
        ros_image = self.br.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.image_pub.publish(ros_image)

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8DynamicObjectAvoidanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

