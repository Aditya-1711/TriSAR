import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
import json
import random
import sys
import numpy as np

sys.path.append('/ros2_ws/src/')
from trishield_core.blackboard import Blackboard
from trishield_core.ga_allocator import HeterogeneousGA
from trishield_core.agent import UAVAgent

class BlackboardNode(Node):
    def __init__(self):
        super().__init__('tri_blackboard_node')
        
        # Publishers and Subscribers
        self.publisher_ = self.create_publisher(String, '/trishield/global_blackboard', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/trishield/mission_markers', 10)
        self.state_sub = self.create_subscription(String, '/trishield/agent_states', self.agent_state_callback, 10)
        
        self.bb = Blackboard()
        self.ga = HeterogeneousGA(self.bb)
        
        # Register static environment over time
        self.bb.register_threat("Trapped_Survivor_RooftopA", [20, 20, 15], "rooftop_rescue")
        self.bb.register_threat("Trapped_Survivor_RooftopB", [15, -15, 10], "rooftop_rescue")
        self.bb.victims["Survivor_1"] = {'pos': [-15, 15, 0], 'urgency': 10}
        
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info("TriShield Dynamic Global Blackboard Online")

    def agent_state_callback(self, msg):
        try:
            state = json.loads(msg.data)
            aid = state['id']
            pos = state['pos']
            vel = state['vel']
            battery = state['battery']
            mission_status = state['mission_status']
            
            # Reconstruct UAV drone agent
            agent = UAVAgent(aid, pos)
            
            agent.pos = pos
            agent.vel = vel
            agent.battery = battery
            agent.mission_status = mission_status
            agent.max_speed = state.get('max_speed', agent.max_speed)
            
            self.bb.broadcast_state(agent)
            
        except Exception as e:
            self.get_logger().warn(f"Bad packet received: {e}")

    def timer_callback(self):
        try:
            # 1. Run Dynamic GA Reassignment based on active drone lifespans
            assignments = self.ga.allocate()
            
            # 2. Package environmental data
            restricted_zones = [
                {'pos': [0, 10, 5], 'radius': 5.0}, # Central obstacle No-Fly zone to force rerouting
                {'pos': [-10, -5, 0], 'radius': 8.0}
            ]
            
            wind_vector = [0.5, 0.2, 0.0] # Slight crosswind drag
            
            state = {
                'threats': self.bb.threats,
                'victims': self.bb.victims,
                'assignments': assignments,
                'restricted_zones': restricted_zones,
                'wind_vector': wind_vector,
                'peer_positions': {aid: (a.pos.tolist() if isinstance(a.pos, np.ndarray) else list(a.pos)) for aid, a in self.bb.agent_states.items()}
            }
            
            msg = String()
            msg.data = json.dumps(state)
            self.publisher_.publish(msg)
            
            # 3. Publish Global Mission Markers (Visuals for Foxglove)
            self.publish_mission_markers()
            
            self.get_logger().info(f'Broadcasting Blackboard - Allocated {len(assignments)} tasks. Seen {len(self.bb.agent_states)} agents.')
        except Exception as e:
            self.get_logger().error(f"Blackboard timer_callback crashed: {e}")

    def publish_mission_markers(self):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        
        # Threats: Red X
        for tid, t in self.bb.threats.items():
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'threats'
            m.id = hash(tid) % 100000
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = [float(c) for c in t['pos']]
            m.scale.x, m.scale.y, m.scale.z = [3.0, 3.0, 3.0]
            m.color.r, m.color.g, m.color.b, m.color.a = [1.0, 0.0, 0.0, 0.8] # Red
            marker_array.markers.append(m)
            
            # Label
            label = Marker()
            label.header.frame_id = 'map'
            label.header.stamp = now
            label.ns = 'threat_labels'
            label.id = m.id + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.pose.position.x, label.pose.position.y, label.pose.position.z = [float(c) for c in t['pos']]
            label.pose.position.z += 4.0
            label.scale.z = 2.0
            label.color.r, label.color.g, label.color.b, label.color.a = [1.0, 1.0, 1.0, 1.0]
            label.text = f"THREAT: {tid}"
            marker_array.markers.append(label)

        # Victims: Orange Sphere
        for vid, v in self.bb.victims.items():
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'victims'
            m.id = hash(vid) % 100000
            m.type = Marker.SPHERE
            m.pose.position.x, m.pose.position.y, m.pose.position.z = [float(c) for c in v['pos']]
            m.scale.x, m.scale.y, m.scale.z = [4.0, 4.0, 4.0]
            m.color.r, m.color.g, m.color.b, m.color.a = [1.0, 0.5, 0.0, 0.8] # Orange
            marker_array.markers.append(m)
            
            # Label
            label = Marker()
            label.header.frame_id = 'map'
            label.header.stamp = now
            label.ns = 'victim_labels'
            label.id = m.id + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.pose.position.x, label.pose.position.y, label.pose.position.z = [float(c) for c in v['pos']]
            label.pose.position.z += 5.0
            label.scale.z = 2.0
            label.color.r, label.color.g, label.color.b, label.color.a = [1.0, 1.0, 1.0, 1.0]
            label.text = f"VICTIM: {vid} (Urgency: {v['urgency']})"
            marker_array.markers.append(label)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = BlackboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
