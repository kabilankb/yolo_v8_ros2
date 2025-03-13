from setuptools import find_packages, setup

package_name = 'yolov8_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'ultralytics',
        'opencv-python',
        'torch',
        'torchvision',
    ],
    zip_safe=True,
    maintainer='kabilankb',
    maintainer_email='kabilankb@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
             'yolov8_node = yolov8_ros2.yolov8_node:main',
             'yolov8_node_segmentation = yolov8_ros2.yolov8_node_segmentation:main',
             'object_tf = yolov8_ros2.object_tf:main',
             'yolov8_avoidance = yolov8_ros2.yolov8_avoidance:main',
             'yolov8_text = yolov8_ros2.yolov8_text:main',
             'isaac_moveit = yolov8_ros2.isaac_moveit:main',
             'moveit_isaac = yolov8_ros2.moveit_isaac:main',
        ],
    },
)

