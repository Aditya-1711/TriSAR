import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    """
    Launch file to spin up Gazebo, spawn 5 PX4 SITL drone models,
    and open RViz for visualization.
    """
    import os
    home_dir = os.path.expanduser("~")
    px4_dir = os.path.join(home_dir, "PX4-Autopilot")
    
    from ament_index_python.packages import get_package_share_directory
    pkg_dir = get_package_share_directory('trishield_hardware_deploy')
    world_file = os.path.join(pkg_dir, 'worlds', 'disaster.sdf')
    
    launch_desc = []
    
    # Add 5 PX4 SITL instances manually
    for i in range(5):
        env_vars = dict(os.environ)
        env_vars['PX4_GZ_MODEL_POSE'] = f"0,{i * 3},0,0,0,0"
        
        if i == 0:
            env_vars['PX4_GZ_WORLD'] = 'disaster_zone'
            # The first instance MUST use make to properly spin up Gazebo Harmonic with PX4's plugins
            launch_desc.append(
                ExecuteProcess(
                    cmd=['make', 'px4_sitl', 'gz_x500'],
                    cwd=px4_dir,
                    env=env_vars,
                    output='screen'
                )
            )
        else:
            # The remaining instances can inject into the running Gazebo world
            env_vars['PX4_SYS_AUTOSTART'] = '4001'
            env_vars['PX4_SIM_MODEL'] = 'gz_x500'
            launch_desc.append(
                ExecuteProcess(
                    cmd=['./build/px4_sitl_default/bin/px4', '-i', str(i), '-d', 'build/px4_sitl_default/etc'],
                    cwd=px4_dir,
                    env=env_vars,
                    output='screen'
                )
            )
        
        
    # Start the Micro-XRCE-DDS Agent to bridge MAVLink to ROS2
    launch_desc.append(
        ExecuteProcess(
            cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
            output='screen'
        )
    )
    
    # Publish static transform for the 'map' frame so RViz can render markers
    launch_desc.append(
        ExecuteProcess(
            cmd=['ros2', 'run', 'tf2_ros', 'static_transform_publisher', '--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch', '0', '--roll', '0', '--frame-id', 'map', '--child-frame-id', 'base_link'],
            output='screen'
        )
    )
    
    # Start the RViz Marker Publisher for TriShield bounds
    launch_desc.append(
        ExecuteProcess(
            cmd=['ros2', 'run', 'trishield_hardware_deploy', 'rviz_marker_publisher'],
            output='screen'
        )
    )
    
    # Start the PX4 Multi Bridge to read drone telemetry and broadcast TF
    launch_desc.append(
        ExecuteProcess(
            cmd=['ros2', 'run', 'trishield_hardware_deploy', 'px4_bridge'],
            output='screen'
        )
    )
    
    # Start RViz2
    launch_desc.append(
        ExecuteProcess(
            cmd=['rviz2'],
            output='screen'
        )
    )
    
    return LaunchDescription(launch_desc)
