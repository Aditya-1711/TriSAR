import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import String
from visualization_msgs.msg import Marker
import json
import sys
import os
import random
import time
import numpy as np

sys.path.append('/ros2_ws/src/')
from trishield_core.agent import UAVAgent, UGVAgent
from trishield_core.pso_optimizer import PSOOptimizer

class AgentNode(Node):
    def __init__(self):
        super().__init__('tri_agent_node')
        
        self.declare_parameter('agent_id', 'UAV_1')
        self.declare_parameter('agent_type', 'UAVAgent')
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('start_z', 10.0)
        
        self.agent_id = self.get_parameter('agent_id').get_parameter_value().string_value
        self.agent_type = self.get_parameter('agent_type').get_parameter_value().string_value
        sx = self.get_parameter('start_x').get_parameter_value().double_value
        sy = self.get_parameter('start_y').get_parameter_value().double_value
        sz = self.get_parameter('start_z').get_parameter_value().double_value
        
        self.get_logger().info(f"Initializing {self.agent_type}: {self.agent_id} at {[sx, sy, sz]}")
        
        if self.agent_type == 'UAVAgent':
            self.math_agent = UAVAgent(self.agent_id, [sx, sy, sz])
            self.c_r, self.c_g, self.c_b = 0.0, 0.0, 1.0 # Blue
        else:
            self.math_agent = UGVAgent(self.agent_id, [sx, sy, sz])
            self.c_r, self.c_g, self.c_b = 0.0, 1.0, 0.0 # Green
            
        self.pso = PSOOptimizer(safe_distance=3.0)
        
        self.global_state_sub = self.create_subscription(String, '/trishield/global_blackboard', self.blackboard_callback, 10)
        self.state_pub = self.create_publisher(String, '/trishield/agent_states', 10)
        self.marker_pub = self.create_publisher(Marker, f'/trishield/foxglove_markers', 10)
        self.battery_marker_pub = self.create_publisher(Marker, f'/trishield/battery_markers', 10)
        
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.current_threats = {}
        self.current_victims = {}
        self.restricted_zones = []
        self.wind_vector = [0.0, 0.0, 0.0]
        self.peer_positions = {}
        self.message_queue = [] # For simulated latency
        
    def blackboard_callback(self, msg):
        # 1. Packet Loss Simulation (5% drop)
        if random.random() < 0.05:
            self.get_logger().debug(f"[{self.agent_id}] Packet loss - ignored blackboard update.")
            return
            
        # 2. Simulated Latency (Queue processing)
        # We append with a timestamp to delay unpacking by 0.5s network lag
        self.message_queue.append({'time': time.time() + 0.5, 'data': msg.data})
            
    def process_queue(self):
        now = time.time()
        to_process = [m for m in self.message_queue if m['time'] <= now]
        self.message_queue = [m for m in self.message_queue if m['time'] > now]
        
        for msg in to_process:
            try:
                state = json.loads(msg['data'])
                self.current_threats = state.get('threats', {})
                self.current_victims = state.get('victims', {})
                self.restricted_zones = state.get('restricted_zones', [])
                self.wind_vector = state.get('wind_vector', [0.0, 0.0, 0.0])
                self.peer_positions = state.get('peer_positions', {})
                
                # Check assignments from GA
                assignments = state.get('assignments', {})
                
                if self.math_agent.mission_status != 'RTB' and self.math_agent.mission_status != 'DEAD':
                    if self.agent_id in assignments.values():
                        # Find which task it is specifically
                        target_pos = None
                        for tid, aid in assignments.items():
                            if aid == self.agent_id:
                                if tid in self.current_threats:
                                    target_pos = self.current_threats[tid]['pos']
                                elif tid in self.current_victims:
                                    target_pos = self.current_victims[tid]['pos']
                        
                        if target_pos:
                            self.math_agent.assigned_task = np.array(target_pos)
                            self.math_agent.mission_status = 'ACTIVE'
                    else:
                        self.math_agent.assigned_task = None
                        self.math_agent.mission_status = 'IDLE'

            except Exception as e:
                pass
                
    def control_loop(self):
        self.process_queue()
        
        # Construct peer array for Swarm Repulsion
        peers = []
        for aid, pos in self.peer_positions.items():
            if aid != self.agent_id:
                dummy = UAVAgent(aid, pos) if 'UAV' in aid else UGVAgent(aid, pos)
                dummy.pos = np.array(pos)
                peers.append(dummy)
                
        # 1. PSO Engine Physics Check (Pass environmental constraints)
        new_vel = self.pso.compute_velocity(
            self.math_agent, 
            all_agents=peers,
            restricted_zones=self.restricted_zones,
            wind_vector=self.wind_vector
        )
        
        # Slow down time explicitly for visualization factor
        self.math_agent.vel = new_vel * 0.1 
        self.math_agent.update_position(0.1) # DT=0.1
        
        # 2. Broadcast Agent State to Blackboard
        state_msg = {
            'id': self.agent_id,
            'type': self.agent_type,
            'pos': self.math_agent.pos.tolist(),
            'vel': self.math_agent.vel.tolist(),
            'battery': float(self.math_agent.battery),
            'mission_status': self.math_agent.mission_status,
            'max_speed': self.math_agent.max_speed
        }
        self.state_pub.publish(String(data=json.dumps(state_msg)))
        
        # 3. Publish Visuals to Foxglove (Inject Hardware Sensor Uncertainty)
        self.publish_markers()

    def publish_markers(self):
        # Simulate Optimal Hardware Sensor drift (+/- 0.05m)
        drift_x = random.gauss(0, 0.05)
        drift_y = random.gauss(0, 0.05)
        drift_z = random.gauss(0, 0.02) if self.agent_type == 'UAVAgent' else 0.0
        
        visual_x = self.math_agent.pos[0] + drift_x
        visual_y = self.math_agent.pos[1] + drift_y
        visual_z = self.math_agent.pos[2] + drift_z

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = self.agent_id
        marker.id = 1 if self.agent_type == 'UAVAgent' else 2
        marker.type = 2 if self.agent_type == 'UAVAgent' else 1
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(visual_x)
        marker.pose.position.y = float(visual_y)
        marker.pose.position.z = float(visual_z)
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 2.0
        marker.scale.y = 2.0
        marker.scale.z = 2.0
        
        marker.color.r = self.c_r
        marker.color.g = self.c_g
        marker.color.b = self.c_b
        marker.color.a = 1.0 
        
        if self.math_agent.mission_status == 'DEAD':
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5 # Grey out dead bots
            
        if self.math_agent.mission_status == 'RTB':
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0 # Orange for RTB
        
        self.marker_pub.publish(marker)
        
        # Battery Text Marker
        battery_marker = Marker()
        battery_marker.header.frame_id = 'map'
        battery_marker.header.stamp = self.get_clock().now().to_msg()
        battery_marker.ns = f"{self.agent_id}_battery"
        battery_marker.id = 100 + (1 if self.agent_type == 'UAVAgent' else 2) 
        # Using hash for unique ID
        battery_marker.id = hash(self.agent_id) % 100000
        battery_marker.type = Marker.TEXT_VIEW_FACING
        battery_marker.action = Marker.ADD
        
        battery_marker.pose.position.x = float(visual_x)
        battery_marker.pose.position.y = float(visual_y)
        battery_marker.pose.position.z = float(visual_z + 2.5) # Float above
        
        battery_marker.scale.z = 1.0 # Text height
        battery_marker.color.r = 1.0
        battery_marker.color.g = 1.0
        battery_marker.color.b = 1.0
        battery_marker.color.a = 1.0
        
        status = self.math_agent.mission_status
        bat = max(0, self.math_agent.battery)
        
        task_label = ""
        if self.math_agent.assigned_task is not None:
             # Find name of current task for visual debugging
             task_label = f"\nTARGET: {self.math_agent.assigned_task}" # Shows coordinates for now
             
        battery_marker.text = f"{self.agent_id}\n{bat:.1f}% [{status}]{task_label}"
        
        self.battery_marker_pub.publish(battery_marker)

def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
