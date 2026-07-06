from setuptools import setup
import os
from glob import glob

package_name = 'trishield_ros2'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        # Include marker file for ament index
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='TriShield Architect',
    maintainer_email='architect@trishield.os',
    description='Decentralized heterogeneous swarm AI logic mapped to ROS 2 Nodes',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tri_agent_node = trishield_ros2.tri_agent_node:main',
            'tri_blackboard_node = trishield_ros2.tri_blackboard_node:main'
        ],
    },
)
