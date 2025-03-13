import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from ultralytics import YOLO

class Yolov8Node(Node):
    def __init__(self):
        super().__init__('yolov8_node')
        
        self.subscription = self.create_subscription(
            Image,
            '/rgb',  # Replace with your actual camera topic
            self.camera_callback,
            10)
        
        # Separate publishers for segmented image and annotated image
        self.segmentation_publisher_ = self.create_publisher(Image, '/yolov8/segmented_image', 10)
        self.detection_publisher_ = self.create_publisher(Image, '/yolov8/detected_image', 10)
        
        self.bridge = CvBridge()
        self.model = YOLO('/home/kabilankb/ultralytics/models/yolov8n-seg.pt')  # Replace with the correct model path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def camera_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image)

        # Get segmentation masks and publish to a separate topic
        masks = results[0].masks

        if masks is not None:
            for mask in masks.data:  # masks.data contains the actual mask tensors
                # Move mask to CPU and convert to NumPy array
                mask_np = mask.cpu().numpy().astype('uint8') * 255
                mask_msg = self.bridge.cv2_to_imgmsg(mask_np, encoding="mono8")
                self.segmentation_publisher_.publish(mask_msg)

        # Draw bounding boxes and labels on the image and publish to a separate topic
        annotated_image = results[0].plot()
        
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
        self.detection_publisher_.publish(annotated_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Yolov8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

