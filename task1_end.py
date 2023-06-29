#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.pytorch_models import *
from pcms.openvino_models import Yolov8, HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist
import math
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest
import time
from mr_voice.msg import Voice
from std_msgs.msg import String
from rospkg import RosPack
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import Imu
from typing import Tuple, List
from RobotChassis import RobotChassis
import datetime


class FollowMe(object):
    def __init__(self) -> None:
        self.pre_x, self.pre_z = 0.0, 0.0

    def find_cx_cy(self) -> Tuple[int, int]:
        max_ = 1
        global up_image, pree_cx, pree_cy, up_depth,net_pose
        cx, cy = 0, 0
        rcx, rcy = 0, 0
        yn = "no"
        l=[5,6,11,12]
        h, w = up_image.shape[:2]
        pose = None
        poses = net_pose.forward(up_image)
        for i, pose in enumerate(poses):
            point = []
            for j, (x, y, preds) in enumerate(pose):  # x: ipex 坐標 y: ipex 坐標 preds: 准度
                if preds <= 0:
                    continue
                x, y = map(int, [x, y])
                _,_,hg=self.get_real_xyz(up_depth,x,y)
                if hg>=2000: continue
                for num in [5, 12]:
                    point.append(j)
            if len(point) == 2:
                pose = poses[i]
                break
        if pose is not None:
            cx5, cy5, cx6, cy6, cx11, cy11, cx12, cy12 = 0, 0, 0, 0, 0, 0, 0, 0
            n1, n2, n3, n4 = 5, 6, 11, 12
            #print(pose)
            cx5, cy5 = self.get_pose_target(pose, n1)
            cx6, cy6 = self.get_pose_target(pose, n2)
            cx11, cy11 = self.get_pose_target(pose, n3)
            cx12, cy12 = self.get_pose_target(pose, n4)

            if cx5 != -1 and cx12 != -1:
                
                ax, ay, az = self.get_real_xyz(up_depth, cx5, cy5)

                
                bx, by, bz = self.get_real_xyz(up_depth, cx12, cy12)
                rcx = (cx12 - cx5) // 2 + cx5
                rcy = (cy12 - cy5) // 2 + cy5
                cv2.circle(up_image, (rcx, rcy), 5, (0, 255, 0), -1)
                cv2.rectangle(up_image, (cx5,cy5), (cx12,cy12), (0, 0,255), 1)
            elif cx6 != -1 and cx11 != -1:

                
                ax, ay, az = self.get_real_xyz(up_depth, cx6, cy6)

                
                bx, by, bz = self.get_real_xyz(up_depth, cx11, cy11)
                
                rcx = (cx11 - cx6) // 2 + cx6
                rcy = (cy11 - cy6) // 2 + cy6
                cv2.circle(up_image, (rcx, rcy), 5, (0, 255, 0), -1)
                #cv2.rectangle(up_image, (cx6,cy6), (cx11,cy11), (0, 0,255), 1)
            else:
                return 0, 0, up_image, "no"
            return rcx, rcy, up_image, "yes"
        return 0, 0, up_image, "no"
    def get_pose_target(self, pose, num):
        p = []
        for i in [num]:
            if pose[i][2] > 0:
                p.append(pose[i])
            
        if len(p) == 0:
            return -1, -1
        return int(p[0][0]), int(p[0][1])
    def get_real_xyz(self, depth, x: int, y: int) -> Tuple[float, float, float]:
        if x < 0 or y < 0:
            return 0, 0, 0

        a = 49.5 * np.pi / 180
        b = 60.0 * np.pi / 180
        d = depth[y][x]
        h, w = depth.shape[:2]
        if d == 0:
            for k in range(1, 15, 1):
                if d == 0 and y - k >= 0:
                    for j in range(x - k, x + k, 1):
                        if not (0 <= j < w):
                            continue
                        d = depth[y - k][j]
                        if d > 0:
                            break
                if d == 0 and x + k < w:
                    for i in range(y - k, y + k, 1):
                        if not (0 <= i < h):
                            continue
                        d = depth[i][x + k]
                        if d > 0:
                            break
                if d == 0 and y + k < h:
                    for j in range(x + k, x - k, -1):
                        if not (0 <= j < w):
                            continue
                        d = depth[y + k][j]
                        if d > 0:
                            break
                if d == 0 and x - k >= 0:
                    for i in range(y + k, y - k, -1):
                        if not (0 <= i < h):
                            continue
                        d = depth[i][x - k]
                        if d > 0:
                            break
                if d > 0:
                    break
        x = x - w // 2
        y = y - h // 2
        real_y = y * 2 * d * np.tan(a / 2) / h
        real_x = x * 2 * d * np.tan(b / 2) / w
        return real_x, real_y, d

    def calc_linear_x(self, cd: float, td: float) -> float:
        if cd <= 0:
            return 0
        e = cd - td
        p = 0.0005
        x = p * e
        if x > 0:
            x = min(x, 0.3)
        if x < 0:
            x = max(x, -0.3)
        return x

    def calc_angular_z(self, cx: float, tx: float) -> float:
        if cx < 0:
            return 0
        e = tx - cx
        p = 0.0025
        z = p * e
        if z > 0:
            z = min(z, 0.2)
        if z < 0:
            z = max(z, -0.2)
        return z

    def calc_cmd_vel(self, image, depth) -> Tuple[float, float]:
        image = image.copy()
        depth = depth.copy()

        cx, cy, frame, yn = self.find_cx_cy()
        if yn == "no":
            cur_x, cur_z = 0, 0
            return cur_x, cur_z,frame,"no"

        print(cx, cy)
        _, _, d = self.get_real_xyz(depth, cx, cy)

        cur_x = self.calc_linear_x(d, 500)
        cur_z = self.calc_angular_z(cx, 320)

        dx = cur_x - self.pre_x
        if dx > 0:
            dx = min(dx, 0.3)
        if dx < 0:
            dx = max(dx, -0.3)

        dz = cur_z - self.pre_z
        if dz > 0:
            dz = min(dz, 0.2)
        if dz < 0:
            dz = max(dz, -0.2)

        cur_x = self.pre_x + dx
        cur_z = self.pre_z + dz

        self.pre_x = cur_x
        self.pre_z = cur_z

        return cur_x, cur_z,frame,"yes"
def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)


def turn_to(angle: float, speed: float):
    global _imu
    max_speed = 1.82
    limit_time = 8
    start_time = rospy.get_time()
    while True:
        q = [
            _imu.orientation.x,
            _imu.orientation.z,
            _imu.orientation.y,
            _imu.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(q)
        e = angle - yaw
        print(yaw, e)
        if yaw < 0 and angle > 0:
            cw = np.pi + yaw + np.pi - angle
            aw = -yaw + angle
            if cw < aw:
                e = -cw
        elif yaw > 0 and angle < 0:
            cw = yaw - angle
            aw = np.pi - yaw + np.pi + angle
            if aw < cw:
                e = aw
        if abs(e) < 0.01 or rospy.get_time() - start_time > limit_time:
            break
        move(0.0, max_speed * speed * e)
        rospy.Rate(20).sleep()
    move(0.0, 0.0)


def turn(angle: float):
    global _imu
    q = [
        _imu.orientation.x,
        _imu.orientation.y,
        _imu.orientation.z,
        _imu.orientation.w
    ]
    roll, pitch, yaw = euler_from_quaternion(q)
    target = yaw + angle
    if target > np.pi:
        target = target - np.pi * 2
    elif target < -np.pi:
        target = target + np.pi * 2
    turn_to(target, 0.1)



def set_gripper(angle, t):
    service_name = "/goal_tool_control"
    rospy.wait_for_service(service_name)

    try:
        service = rospy.ServiceProxy(service_name, SetJointPosition)

        request = SetJointPositionRequest()
        request.joint_position.joint_name = ["gripper"]
        request.joint_position.position = [angle]
        request.path_time = t

        response = service(request)
        return response
    except Exception as e:
        rospy.loginfo("%s" % e)
        return False


def open_gripper(t):
    return set_gripper(0.01, t)


def close_gripper(t):
    return set_gripper(-0.01, t)


def get_pose_target(pose, num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])

    if len(p) == 0:
        return -1, -1
    return int(p[0][0]), int(p[0][1])



def get_real_xyz(dp, x, y, num):
    a1=49.5
    b1=60.0
    if num == 2:
        a1=55.0
        b1=86.0
    a = a1 * np.pi / 180
    b = b1 * np.pi / 180
    d = dp[y][x]
    h, w = dp.shape[:2]
    if d == 0:
        for k in range(1, 15, 1):
            if d == 0 and y - k >= 0:
                for j in range(x - k, x + k, 1):
                    if not (0 <= j < w):
                        continue
                    d = dp[y - k][j]
                    if d > 0:
                        break
            if d == 0 and x + k < w:
                for i in range(y - k, y + k, 1):
                    if not (0 <= i < h):
                        continue
                    d = dp[i][x + k]
                    if d > 0:
                        break
            if d == 0 and y + k < h:
                for j in range(x + k, x - k, -1):
                    if not (0 <= j < w):
                        continue
                    d = dp[y + k][j]
                    if d > 0:
                        break
            if d == 0 and x - k >= 0:
                for i in range(y + k, y - k, -1):
                    if not (0 <= i < h):
                        continue
                    d = dp[i][x - k]
                    if d > 0:
                        break
            if d > 0:
                break

    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d


def set_joints(joint1, joint2, joint3, joint4, t):
    service_name = "/goal_joint_space_path"
    rospy.wait_for_service(service_name)

    try:
        service = rospy.ServiceProxy(service_name, SetJointPosition)

        request = SetJointPositionRequest()
        request.joint_position.joint_name = [
            "joint1", "joint2", "joint3", "joint4"]
        request.joint_position.position = [joint1, joint2, joint3, joint4]
        request.path_time = t

        response = service(request)
        return response
    except Exception as e:
        rospy.loginfo("%s" % e)
        return False
def move_to(x, y, z, t):
    service_name = "/goal_task_space_path_position_only"
    rospy.wait_for_service(service_name)
    
    try:
        service = rospy.ServiceProxy(service_name, SetKinematicsPose)
        
        request = SetKinematicsPoseRequest()
        request.end_effector_name = "gripper"
        request.kinematics_pose.pose.position.x = x
        request.kinematics_pose.pose.position.y = y
        request.kinematics_pose.pose.position.z = z
        request.path_time = t
        
        response = service(request)
        return response
    except Exception as e:
        rospy.loginfo("%s" % e)
        return False
def say(a):
    global publisher_speaker
    publisher_speaker.publish(a)

#callback
def callback_imu(msg):
    global _imu
    _imu = msg
def callback_voice(msg):
    global s
    s = msg.text
#astrapro
def callback_image1(msg):
    global _image1
    _image1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")
def callback_depth1(msg):
    global _depth1
    _depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")
#gemini2
def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")
def callback_depth(msg):
    global _depth
    _depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")

if __name__ == "__main__":    
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")

    #main
    print("astra rgb")
    _image1 = None
    _topic_image1 = "/cam2/rgb/image_raw"
    rospy.Subscriber(_topic_image1, Image, callback_image1)
    
    print("astra depth")
    _depth1 = None
    _topic_depth1 = "/cam2/depth/image_raw"
    rospy.Subscriber(_topic_depth1, Image, callback_depth1)

    print("gemini2 rgb")
    _frame = None
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image)

    print("gemini2 depth")
    _depth= None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth)

    s=""
    print("cmd_vel")
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    print("speaker")
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)

    print("arm")
    t=3.0
    open_gripper(t)
    
    print("yolov8")
    Kinda = np.loadtxt(RosPack().get_path("mr_dnn") + "/Kinda.csv")
    dnn_yolo = Yolov8("bagv4")
    dnn_follow = Yolov8("yolov8n")
    dnn_yolo.classes = ['obj']

    print("pose")
    net_pose = HumanPoseEstimation()

    print("waiting imu")
    topic_imu = "/imu/data"
    _imu=None
    rospy.Subscriber(topic_imu, Imu, callback_imu)
    rospy.wait_for_message(topic_imu, Imu)

    print("chassis")
    chassis = RobotChassis()
    
    _fw = FollowMe()

    print("finish loading, start")
    h,w,c = _image1.shape
    img = np.zeros((h,w*2,c),dtype=np.uint8)
    img[:h,:w,:c] = _image1
    img[:h,w:,:c] = _frame
    
    # u_var
    d, one, mask, key, is_turning = 1, 0, 0, 0, False
    ax, ay, az, bx, by, bz = 0, 0, 0, 0, 0, 0
    pre_x, pre_z, haiya, bruh, lcnt, rcnt, run, p_cnt, focnt = 0.0, 0.0, 0, 0, 0, 0, 0, 0, 1
    pos, cnt_list = [2.77, 1.82, 0.148], []
    pre_s=""
    # main var
    t, ee, s = 3.0, "", ""
    step="get_bag"
    action="none"
    move_turn="none"
    # wait for prepare
    print("start")
    time.sleep(10)
    
    lr="middle"
    # var in camera
    px, py, pz, pree_cx, pree_cy,= 0,0,0,0,0
    posecnt=0
    move_turn="turn"
    #senser var
    class_need=0
    move_to(0.287,0,0.193,3.0)
    time.sleep(2)
    move_to(0.30,0.019,0.0,3.0)
    time.sleep(3)
    move_to(0.25,0.019,0.0,3.0)
    time.sleep(2)
    move_to(0.20,0.019,0.0,3.0)
    time.sleep(3)
    while not rospy.is_shutdown():
        #voice check
        print("in")
        if s!="" and s!=pre_s:
            print(s)
            pre_s = s
        
        rospy.Rate(10).sleep()
        if _frame is None: print("gemini2 rgb none")
        if _depth is None: print("gemini2 depth none")
        if _depth1 is None: print("astra depth none")
        if _image1 is None: print("astra rgb none")
        
        if _depth is None or _image1 is None or _depth1 is None or _frame is None: continue
        
        #var needs in while
        cx1,cx2,cy1,cy2=0,0,0,0
        detection_list=[]

        down_image = _frame.copy()
        down_depth = _depth.copy()
        up_image= _image1.copy()
        up_depth= _depth1.copy()
        if step=="get_bag":
            #yolov8 detect
            detections = dnn_yolo.forward(down_image)[0]["det"]
            for i, detection in enumerate(detections):
                #print(detection)
                x1, y1, x2, y2, score, class_id = map(int, detection)
                score = detection[4]
                cx = (x2 - x1) // 2 + x1
                cy = (y2 - y1) // 2 + y1
                if score > 0.55 and class_id == class_need:
                    detection_list.append([x1,y1,x2,y2,cx,cy])
                    cv2.rectangle(down_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(down_image, (cx, cy), 5, (0, 255, 0), -1)
            #pose detect
            pose=None
            poses = net_pose.forward(up_image)
            for i, pose in enumerate(poses):
                point = []
                for j, (x,y,preds) in enumerate(pose): #x: ipex 坐標 y: ipex 坐標 preds: 准度
                    if preds <= 0: continue
                    x,y = map(int,[x,y])
                    for num in [8,10]:
                        point.append(j)
                if len(point) == 2:
                    pose = poses[i]
                    break
        
        #if step=="none": continue
        if step=="get_bag":
            if pose is not None:
                cx7, cy7, cx9, cy9, cx5, cy5 = 0, 0, 0, 0, 0, 0
                n1, n2, n3 = 6, 8, 10
                #print(pose)
                cx7, cy7 = get_pose_target(pose, n2)

                cx9, cy9 = get_pose_target(pose, n3)

                cx5, cy5 = get_pose_target(pose, n1)
                
                
                if cx7 == -1 and cx9 != -1:
                    cv2.circle(up_image, (cx5, cy5), 5, (0, 255, 0), -1)
                    ax, ay, az = get_real_xyz(up_depth,cx5, cy5,1)

                    cv2.circle(up_image, (cx9, cy9), 5, (0, 255, 0), -1)
                    bx, by, bz = get_real_xyz(up_depth,cx9, cy9,1)
                elif cx7 != -1 and cx9 == -1:

                    cv2.circle(up_image, (cx5, cy5), 5, (0, 255, 0), -1)
                    ax, ay, az = get_real_xyz(up_depth,cx5, cy5,1)

                    cv2.circle(up_image, (cx7, cy7), 5, (0, 255, 0), -1)
                    bx, by, bz = get_real_xyz(up_depth,cx7, cy7,1)
                elif cx7 == -1 and cx9 == -1:
                    pass
                else:
                    cv2.circle(up_image, (cx7, cy7), 5, (0, 255, 0), -1)
                    ax, ay, az = get_real_xyz(up_depth,cx7, cy7,1)

                    cv2.circle(up_image, (cx9, cy9), 5, (0, 255, 0), -1)
                    bx, by, bz = get_real_xyz(up_depth,cx9, cy9,1)
                if len(detection_list) < 1: 
                    print("no bag")
                    
                if len(detection_list)==2:
                    print(detection_list)
                    if len(detection_list)==2:
                        sort_detection=sorted(detection_list, key=(lambda x:x[0]))
                        if posecnt==0:
                            if ax<0: lr="left"
                            else: lr="right"
                            posecnt+=1
                        if lr=="left": need_position=sort_detection[0]
                        else: need_position=sort_detection[1]
                    
                    print(need_position)
                    print("ho")
                    x1, y1, x2, y2, cx2, cy2 = map(int, need_position)
                    cv2.rectangle(down_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(down_image, (cx2, cy2), 5, (0, 0, 255), -1)

                    #capture evidence
                    now = datetime.datetime.now()
                    filename = now.strftime("%Y-%m-%d_%H-%M-%S.jpg")
                    output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                    cv2.imwrite(output_dir + filename, down_image)
                    if move_turn=="turn":
                        
                        h,w,c = down_image.shape
                        x1, y1, x2, y2, cx2, cy2 = map(int, need_position)
                        e = w//2 - cx2
                        v = 0.001 * e
                        if v > 0:
                            v = min(v, 0.3)
                        if v < 0:
                            v = max(v, -0.3)
                        move(0, v)
                        print(e)
                        if abs(e) <= 5:
                            say("walk")
                            action="front"
                            move_turn="none"
                            step="none"
                            print("ys")
        
        if step=="check_voice":
             if "thank" in s or "stop" in s or "now" in s or "Thank" in s or "Stop" in s or "THANK" in s or "STOP" in s or "NOW" in s or "you" in s or "You" in s:
                action="none"
                say("I will go back now, bye bye")
                joint1, joint2, joint3, joint4 = 0.000, 0.0, 0, 1.2
                set_joints(joint1, joint2, joint3, joint4, t)
                time.sleep(t)
                open_gripper(t)
                time.sleep(3)
                close_gripper(t)
                time.sleep(3)
                open_gripper(t)
                time.sleep(3)
                close_gripper(t)
                time.sleep(3)
                joint1, joint2, joint3, joint4 = 0.000, -1.0, 0.3,0.70
                set_joints(joint1, joint2, joint3, joint4, t)
                
                time.sleep(2.5)
                action="back"
                step="none"

        if action=="front":
            print("front")
            cx, cy = w // 2, h // 2
            for i in range(cy + 1, h):
                if _depth[cy][cx] == 0 or 0 < _depth[i][cx] < _depth[cy][cx]:
                    cy = i 
            _,_,d = get_real_xyz(_depth,cx,cy,2)
            while d > 0 or abs(e) >= 10:
                _,_,d1 = get_real_xyz(_depth,cx,cy,2)
                e = d1 - 400 #number is he last distance
                if e<=10:
                    break
                v = 0.001 * e
                if v > 0:
                    v = min(v, 0.2)
                if v < 0:
                    v = max(v, -0.2)
                print(d1, e, v)
                move(v, 0)
            print("got there")
            action="grap"
            move_turn="none"
            step="none"
            
        if action=="grap":
            move_to(0.20,-0.03,0.0,3.0)
            time.sleep(3)
            for i in range(900): move(0.2,0)
            say("I get it")
            time.sleep(t)
            close_gripper(t)
            time.sleep(t)
            move_to(0.30,-0.03,0.0,3.0)
            time.sleep(3)
            
            joint1, joint2, joint3, joint4 = -0.106, 0.419, 0.365, -1.4
            set_joints(joint1, joint2, joint3, joint4, t)
            time.sleep(t)
            joint1, joint2, joint3, joint4 = 0,0,0, -1.0
            set_joints(joint1, joint2, joint3, joint4, t)
            time.sleep(t)
            time.sleep(3)
            say("I will follow you now")
            for i in range(50000): move(-0.2,0)
            action="follow"
            
            
            
        
        if action=="follow":
            print('follow')
            msg=Twist()
            x, z, up_image,yn = _fw.calc_cmd_vel(up_image, up_depth)
            print("turn_x_z:", x, z)
            if yn=="no":
                x,z=0,0
                say("slower")
            move(x,z)
            step="check_voice"
        if action == "back":
            chassis.move_to(6.73,-2.59,0.00171)
            #checking
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
            time.sleep(1)
            chassis.move_to(6.78,-2.68,-0.00129)
            #checking
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
            time.sleep(1)
            chassis.move_to(6.39,-5.1,-0.00173)
            #checking
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
            time.sleep(1)
            chassis.move_to(-3.29,-4.12,0.00359)
            #checking
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
            time.sleep(1)
            break
        
        print("show")
        h,w,c = up_image.shape
        upout=cv2.line(up_image, (320,0), (320,500), (0,255,0), 5)
        downout=cv2.line(down_image, (320,0), (320,500), (0,255,0), 5)
        img = np.zeros((h,w*2,c),dtype=np.uint8)
        img[:h,:w,:c] = upout
        img[:h,w:,:c] = downout
        
        cv2.imshow("frame", img)   
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
