import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from math import pi, cos, sin

# 机械臂各关节的初始角度
joint_angles = [0, 0, 0, 0, 0]

# 物体的真实坐标
object_x = 0.3
object_y = 0
object_z = 0.2

# 机械臂末端执行器到物体的距离和姿态
distance = 0.2
yaw = 0
pitch = pi / 2
roll = 0

# 计算机械臂末端执行器的坐标
x = object_x - distance * cos(pitch) * sin(yaw)
y = object_y + distance * cos(pitch) * cos(yaw)
z = object_z + distance * sin(pitch)

# 将坐标转化为目标点
goal_position = [x, y, z]
goal_orientation = [0, 0, 0, 1]  # 四元数表示姿态

# 计算逆运动学
from open_manipulator_msgs.srv import InverseKinematics

rospy.wait_for_service('/open_manipulator/goal_joint_space_path')
inverse_kinematics = rospy.ServiceProxy('/open_manipulator/goal_joint_space_path', InverseKinematics)
response = inverse_kinematics(goal_position, goal_orientation, joint_angles)
joint_angles = response.goal_joint_state.position

