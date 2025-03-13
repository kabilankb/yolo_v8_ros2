import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM

class YOLOv8Node(Node):
    def __init__(self):
        super().__init__('yolov8_node')
        self.publisher_ = self.create_publisher(Image, 'yolov8/image', 10)
        self.model = YOLO('/home/kabilankb/ultralytics/models/yolov8n.pt')
        self.br = CvBridge()
        
        # Initialize text generation model
        self.tokenizer = AutoTokenizer.from_pretrained("google/tinybert")
        self.text_model = AutoModelForCausalLM.from_pretrained("google/tinybert")

        # Subscriber to the camera topic
        self.subscriber_ = self.create_subscription(
            Image,
            '/jetson_webcam',  # Update this with your camera topic
            self.camera_callback,
            10
        )

    def camera_callback(self, msg):
        try:
            # Check the encoding of the incoming image
            if msg.encoding == '8UC3':
                # If the encoding is '8UC3', no conversion is necessary, but you can still convert it to BGR
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            else:
                # Convert the ROS Image message to an OpenCV image
                cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform object detection
            results = self.model(cv_image)

            # Draw detections on the image
            object_labels = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

                for box, class_id in zip(boxes, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = self.model.names[int(class_id)]  # Get the class name

                    # Draw the bounding box
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Put the class name above the bounding box
                    cv2.putText(cv_image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    object_labels.append(class_name)

            # Generate a description from detected objects
            description = self.generate_description(object_labels)

            # Optionally print or use the description
            self.get_logger().info(f'Description: {description}')

            # Convert the OpenCV image back to a ROS Image message
            ros_image = self.br.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.publisher_.publish(ros_image)
        
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')
        except Exception as e:
            self.get_logger().error(f'Error during processing: {e}')

    def generate_description(self, object_labels):
        if not object_labels:
            return "No objects detected."

        input_text = f"The image contains: {', '.join(object_labels)}."
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")

        # Generate description
        outputs = self.text_model.generate(inputs, max_length=50)
        description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return description

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

