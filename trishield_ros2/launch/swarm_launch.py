from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    nodes = [
        # 1. Start WebSockets Backbone for 3D UI Streaming
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen'
        ),
        
        # 2. Start the Distributed Network Backbone (Now with GA allocation logic embedded)
        Node(
            package='trishield_ros2',
            executable='tri_blackboard_node',
            name='global_blackboard',
            output='screen'
        )
    ]
    
    # 3. Deploy Configurable Drones dynamically
    UAV_COUNT = 5
    UGV_COUNT = 3
    
    for i in range(1, UAV_COUNT + 1):
        # Evenly space them along the X axis, shifted slightly back
        start_x = float((i - UAV_COUNT/2) * 4.0)
        start_y = -10.0
        start_z = 10.0 + float(i) # Stagger heights to prevent initial jitter
        
        nodes.append(
            Node(
                package='trishield_ros2',
                executable='tri_agent_node',
                name=f'uav_{i}_controller',
                parameters=[{
                    'agent_id': f'UAV_{i}', 
                    'agent_type': 'UAVAgent',
                    'start_x': start_x,
                    'start_y': start_y,
                    'start_z': start_z,
                }],
                output='screen'
            )
        )
        
    for i in range(1, UGV_COUNT + 1):
        # Spawn UGVs on the other side
        start_x = float((i - UGV_COUNT/2) * 5.0)
        start_y = 10.0
        start_z = 0.0
        
        nodes.append(
            Node(
                package='trishield_ros2',
                executable='tri_agent_node',
                name=f'ugv_{i}_controller',
                parameters=[{
                    'agent_id': f'UGV_{i}', 
                    'agent_type': 'UGVAgent',
                    'start_x': start_x,
                    'start_y': start_y,
                    'start_z': start_z,
                }],
                output='screen'
            )
        )
        
    return LaunchDescription(nodes)
