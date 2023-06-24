#!/usr/bin/env python3
from typing import Tuple, List
import numpy as np

class FollowMe(object):
    def __init__(self) -> None:
        self.pre_x, self.pre_z = 0.0, 0.0

    def find_cx_cy(self) -> Tuple[int, int]:
        max_=1
        global dnn_yolo, _image, pree_cx,pree_cy
        cx,cy=0,0
        rcx,rcy=0,0
        frame = _image.copy()
        detections = dnn_yolo.forward(frame)[0]["det"]
        for i, detection in enumerate(detections):
            print(detection)
            x1, y1, x2, y2, score, class_id = map(int, detection)
            score = detection[4]
            print(x1, y1, x2, y2, score, class_id)
            if score > 0.5 and class_id == 0:
                #dnn_yolo.draw_bounding_box(detection, frame)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cx = (x2 - x1) // 2 + x1
                cy = (y2 - y1) // 2 + y1
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, "person", (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                hh=x2-x1
                ll=y2-y1
                if ll*hh>max_:
                    max_=ll*hh
                    rcx,rcy=cx,cy
        
        if rcx!=0:
            pree_cx,pree_cy=rcx,rcy
                
            return rcx,rcy,frame
        else:
            return pree_cx,pree_cy,frame


    def get_real_xyz(self, depth, x: int, y: int) -> Tuple[float, float, float]:
        if x < 0 or y < 0: return 0, 0, 0

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
                        d = deppth[i][x - k]
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
        if cd <= 0: return 0
        e = cd - td
        p = 0.0005
        x = p * e
        if x > 0: x = min(x, 0.5)
        if x < 0: x = max(x, -0.5)
        return x

    def calc_angular_z(self, cx: float, tx: float) -> float:
        if cx < 0: return 0
        e = tx - cx
        p = 0.0025
        z = p * e
        if z > 0: z = min(z, 0.3)
        if z < 0: z = max(z, -0.3)
        return z

    def calc_cmd_vel(self, image, depth) -> Tuple[float, float]:
        image = image.copy()
        depth = depth.copy()
        
        cx, cy, frame = self.find_cx_cy()

        _, _, d = self.get_real_xyz(depth, cx, cy)

        cur_x = self.calc_linear_x(d, 800)
        cur_z = self.calc_angular_z(cx, 320)

        dx = cur_x - self.pre_x
        if dx > 0: dx = min(dx, 0.04)
        if dx < 0: dx = max(dx, -0.04)
        
        dz = cur_z - self.pre_z
        if dz > 0: dz = min(dz, 0.1)
        if dz < 0: dz = max(dz, -0.1)

        cur_x = self.pre_x + dx
        cur_z = self.pre_z + dz

        self.pre_x = cur_x 
        self.pre_z = cur_z 

        return cur_x, cur_z, frame
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
    _topic_image = "/camera/rgb/image_raw"
    rospy.Subscriber(_topic_image, Image, callback_image)
    rospy.wait_for_message(_topic_image, Image)
    
    # Depth Image Subscriber
    _depth = None
    _topic_depth = "/camera/depth/image_raw"
    rospy.Subscriber(_topic_depth, Image, callback_depth)
    rospy.wait_for_message(_topic_depth, Image)
    
    # cmd_vel Publisher
    _msg_cmd = Twist()
    _pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    #Kinda = np.loadtxt(RosPack().get_path("mr_dnn") + "/Kinda.csv")
    print("load")
    dnn_yolo = Yolov8("yolov8n")
    print("f")
    # Models
    #_net_pose = HumanPoseEstimation()
    #detections = dnn_yolo.forward(_image)[0]["det"]
    # Functions
    _fw = FollowMe()
    # Main loop
    pree_cx=0
    pree_cy=0
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        image = _image.copy()
        depth = _depth.copy()
        
        #res = dnn_yolo.forward(image)[0]["det"]
        #print(res)
        #cx, cy, frame = _fw.find_cx_cy()
        x, z, frame = _fw.calc_cmd_vel(image, depth)
        print(x, z)
        #print(cx, cy)
        #poses = _net_pose.forward(image)
        #x, z = _fw.calc_cmd_vel(image, depth, poses)

        _msg_cmd.linear.x = x 
        _msg_cmd.angular.z = z
        _pub_cmd.publish(_msg_cmd)
        
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
        
    rospy.loginfo("FollowMe end!")
