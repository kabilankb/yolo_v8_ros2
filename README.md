# YOLOv8 for Obstacle Avoidance in Isaac Sim

## Introduction
YOLOv8, the latest iteration of the YOLO family, offers improved speed and accuracy, making it ideal for real-time robotics applications. In this blog, we explore how YOLOv8 enables obstacle avoidance in robots and provide a step-by-step guide to implementation.
![Image](https://github.com/user-attachments/assets/1a5102e1-a276-4912-8d55-1a6c7bed442d)
## YOLOv8 for Robotics
YOLOv8 efficiently detects objects, including static and dynamic obstacles, allowing robots to make informed navigation decisions. Its optimized architecture ensures rapid processing, essential for real-time obstacle avoidance.

## Methodology
1. **Camera Feed Acquisition**: Captures the environment.
2. **Object Detection**: YOLOv8 detects objects and assigns bounding boxes.
3. **Distance Estimation**: Determines object proximity.
4. **Obstacle Avoidance**: Adjusts movement based on detected obstacles.
5. **Velocity Command**: Generates appropriate navigation commands.

## Implementation Steps
1. **Initialization**: Sets up publishers and subscribers for image and velocity commands.
2. **Image Processing**: Converts ROS Image messages to OpenCV format.
3. **Object Detection**: Extracts bounding boxes and class IDs.
4. **Distance Calculation**: Uses bounding box size to estimate distance.
5. **Obstacle Avoidance**: Determines movement decisions based on object position.
6. **Velocity Command**: Publishes `Twist` messages to the `cmd_vel` topic.

## Mathematical Basis
- **Distance Estimation**: Uses bounding box height as a distance proxy.
- **Position Calculation**: Determines object center relative to the image center to guide movement.

## Conclusion
YOLOv8 enhances robotic navigation by enabling real-time obstacle detection and avoidance. This ROS2 node implementation demonstrates its effectiveness in ensuring safe and efficient robotic movement.

## Repository
[GitHub Repository](git@github.com:kabilankb/yolo_v8_ros2.git)

## Running the Nodes
```bash
ros2 run yolo_v8_ros2 yolov8_avoidance
ros2 run yolo_v8_ros2 yolov8_node_segmentation
```

