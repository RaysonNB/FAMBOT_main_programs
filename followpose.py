#!/usr/bin/env python3
from typing import Tuple, List
import numpy as np
from mr_voice.msg import Voice
from std_msgs.msg import String
import datetime
import time
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
def say(a):
    global publisher_speaker
    publisher_speaker.publish(a)

if __name__ == "__main__":
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    import cv2
    from pcms.openvino_models import HumanPoseEstimation, Yolov8
    import numpy as np
    from geometry_msgs.msg import Twist


    def callback_image(msg):
        global _image
        _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        

    def callback_depth(msg):
        global _depth
        _depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    

    rospy.init_node("FollowMe")
    rospy.loginfo("FollowMe started!")
    
    # RGB Image Subscriber
    _image = None
    _topic_image = "/cam2/rgb/image_raw"
    rospy.Subscriber(_topic_image, Image, callback_image)
    rospy.wait_for_message(_topic_image, Image)
    
    # Depth Image Subscriber
    _depth = None
    _topic_depth = "/cam2/depth/image_raw"
    rospy.Subscriber(_topic_depth, Image, callback_depth)
    rospy.wait_for_message(_topic_depth, Image)
    net_pose = HumanPoseEstimation(device_name="GPU")
    # cmd_vel Publisher
    _msg_cmd = Twist()
    _pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    #Kinda = np.loadtxt(RosPack().get_path("mr_dnn") + "/Kinda.csv")
    print("load")
    print("f")
    # Models
    _net_pose = HumanPoseEstimation()
    #detections = dnn_yolo.forward(_image)[0]["det"]
    # Functions
    _fw = FollowMe()
    # Main loop
    pree_cx=0
    pree_cy=0
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        up_image = _image.copy()
        up_depth = _depth.copy()
        
        #res = dnn_yolo.forward(image)[0]["det"]
        #print(res)
        #cx, cy, frame = _fw.find_cx_cy()
        x, z, up_image,yn = _fw.calc_cmd_vel(up_image, up_depth)
        print(x, z)
        #print(cx, cy)
        #poses = _net_pose.forward(image)
        #x, z = _fw.calc_cmd_vel(image, depth, poses)
        #publisher_speaker.publish(p)
        if yn=="no":
            x,z=0,0
            say("slower")
        _msg_cmd.linear.x = x 
        _msg_cmd.angular.z = z
        _pub_cmd.publish(_msg_cmd)
        now = datetime.datetime.now()
        filename = now.strftime("%Y-%m-%d_%H-%M-%S.jpg")
        output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
        cv2.imwrite(output_dir + filename, up_image)
        cv2.imshow("frame", up_image)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
        
    rospy.loginfo("FollowMe end!")
