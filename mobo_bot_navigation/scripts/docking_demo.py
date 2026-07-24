#! /usr/bin/env python3
# Copyright 2024 Open Navigation LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import DockRobot, UndockRobot
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from tf_transformations import euler_from_quaternion, quaternion_from_euler

class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3


class DockingTester(Node):

    def __init__(self):
        super().__init__(node_name='docking_tester')
        self.goal_handle = None
        self.result_future = None
        self.status = None
        self.feedback = None

        self.tracked_dock_pose = None

        self.docking_client = ActionClient(self, DockRobot,
                                            'dock_robot')
        self.undocking_client = ActionClient(self, UndockRobot,
                                            'undock_robot')
        
    #Attemp to use tracked pose directly from the camera, uncommment to subscribe to tracked pose

    #     self.dock_pose = self.create_subscription(
    #          PoseStamped, 'detected_dock_pose', self.dock_callback, 10)

    # def dock_callback(self, msg):
    #      print("Received dock pose: " + str(msg))
    #      self.tracked_dock_pose = msg


    def rotate_pose_yaw(self, pose_msg, yaw_offset_rad):
        # Get current orientation as Euler angles
        q = [pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w]
        roll, pitch, yaw = euler_from_quaternion(q)
        
        # Add the offset
        new_yaw = yaw + yaw_offset_rad
        
        # Convert back to quaternion
        new_q = quaternion_from_euler(roll, pitch, new_yaw)
        pose_msg.pose.orientation.x = new_q[0]
        pose_msg.pose.orientation.y = new_q[1]
        pose_msg.pose.orientation.z = new_q[2]
        pose_msg.pose.orientation.w = new_q[3]
        return pose_msg
    
            

    def destroy_node(self):
        self.docking_client.destroy()
        self.undocking_client.destroy()
        super().destroy_node()


 
    def dockRobot(self, dock_pose= None , dock_id = "home_dock", dock_type = "nova_carter_dock", stage = True):
        """Send a `DockRobot` action request."""
        print("Waiting for 'DockRobot' action server")
        while not self.docking_client.wait_for_server(timeout_sec=1.0):
            print('"DockRobot" action server not available, waiting...')

        goal_msg = DockRobot.Goal()
        goal_msg.navigate_to_staging_pose = stage  # if want to navigate before staging

        if dock_pose is not None:
            # Use the supplied pose
            goal_msg.use_dock_id = False
            goal_msg.dock_pose = dock_pose
            goal_msg.dock_type = dock_type
            print('Docking at pose: ' + str(dock_pose) + '...')
        else:
            # Use the dock ID from the database
            goal_msg.use_dock_id = True
            goal_msg.dock_id = dock_id
            print(f'Docking at ID: {dock_id}...')

        
        send_goal_future = self.docking_client.send_goal_async(goal_msg,
                                                                self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            print('Docking request was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    
    def undockRobot(self, dock_type):
        """Send a `UndockRobot` action request."""
        print("Waiting for 'UndockRobot' action server")
        while not self.undocking_client.wait_for_server(timeout_sec=1.0):
            print('"UndockRobot" action server not available, waiting...')

        goal_msg = UndockRobot.Goal()
        goal_msg.dock_type = dock_type

        print('Undocking from dock of type: ' + str(dock_type) + '...')
        send_goal_future = self.undocking_client.send_goal_async(goal_msg,
                                                                 self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            print('Undocking request was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            # task was cancelled or completed
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                print(f'Task with failed with status code: {self.status}')
                return True
        else:
            # Timed out, still processing, not complete yet
            return False

        print('Task succeeded!')
        return True

    def _feedbackCallback(self, msg):
        self.feedback = msg.feedback
        return
    
    def cancelAction(self):
        self.goal_handle.cancel_goal()

    def getFeedback(self):
        """Get the pending action feedback message."""
        return self.feedback

    def getResult(self):
        """Get the pending action result message."""
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN

    def startup(self, node_name='docking_server'):
        # Waits for the node within the tester namespace to become active
        print(f'Waiting for {node_name} to become active..')
        node_service = f'{node_name}/get_state'
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            print(f'{node_service} service not available, waiting...')

        req = GetState.Request()
        state = 'unknown'
        while state != 'active':
            print(f'Getting {node_name} state...')
            future = state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                state = future.result().current_state.label
                print(f'Result of get_state: {state}')
            time.sleep(2)
        return



def main():
    rclpy.init()

    tester = DockingTester()
    tester.startup()

    while True:
        time.sleep(1)

        # # set dock pose
        dock_pose = PoseStamped()
        dock_pose.header.stamp = tester.get_clock().now().to_msg()
        dock_pose.header.frame_id = "map"
        dock_pose.pose.position.x = -0.2
        dock_pose.pose.position.y = 3.0
        

        # # Set the yaw angle (in radians). For example, 90° = π/2 ≈ 1.5708
        yaw_rad = 1.57  # 90 degrees anti clockwise
        dock_pose = tester.rotate_pose_yaw(dock_pose, yaw_rad)  # Rotate by yaw_rad radians 


        #Attemp to use tracked pose directly from the camera, Wait for the first tracked pose
        # print("Waiting for dock pose on topic 'detected_dock_pose'...")
        # while tester.tracked_dock_pose is None:
        #     rclpy.spin_once(tester, timeout_sec=0.1)
        # print("Received dock pose, proceeding with docking.")

        # ---- Dock ----
        # dock_pose = tester.tracked_dock_pose
        # print(tester.tracked_dock_pose.header.frame_id)
        #dock_pose = tester.rotate_pose_yaw(dock_pose, yaw_rad)  # Rota
        

        dock_id = 'flex_dock1'
        
        tester.dockRobot(dock_pose = dock_pose, stage = True) #change dock_pose = dock_pose to dock_id = 'dock_id' if you want to use the dock ID instead of the pose

        # Test cancel action
        # time.sleep(0.5)
        # tester.cancelAction()

        i = 0
        while not tester.isTaskComplete():
            i = i + 1
            if i % 5 == 0:
                print('Docking in progress...')
            time.sleep(1)

        # Do something depending on the return code
        result = tester.getResult()
        if result == TaskResult.SUCCEEDED:
            print('Docking succeeded!')
        elif result == TaskResult.CANCELED:
            print('Docking canceled!')
        elif result == TaskResult.FAILED:
            print('Docking failed!')
        else:
            print('Docking has an invalid return status!')

        time.sleep(3)

        # Undock from this dock
        dock_type = "nova_carter_dock"
        tester.undockRobot(dock_type)

        i = 0
        while not tester.isTaskComplete():
            i = i + 1
            if i % 5 == 0:
                print('Undocking in progress...')
            time.sleep(1)

        # Do something depending on the return code
        result = tester.getResult()
        if result == TaskResult.SUCCEEDED:
            print('Undock succeeded!')
        elif result == TaskResult.CANCELED:
            print('Undock canceled!')
        elif result == TaskResult.FAILED:
            print('Undock failed!')
        else:
            print('Undock has an invalid return status!')


if __name__ == '__main__':
    main()