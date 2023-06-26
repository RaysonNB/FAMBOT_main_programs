#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import Yolov8,HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist
from pcms.pytorch_models import *
from pcms.openvino_yolov8 import *
import math
import time
from mr_voice.msg import Voice
from std_msgs.msg import String
#gemini2
def callback_image2(msg):
    global frame2
    frame2 = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth2(msg):
    global depth2
    depth2 = CvBridge().imgmsg_to_cv2(msg, "passthrough")
#astra
def callback_image1(msg):
    global frame1
    frame1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth1(msg):
    global depth1
    depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    
def get_real_xyz(dp,x, y):
    a = 55.0 * np.pi / 180
    b = 86.0 * np.pi / 180
    d = dp[y][x]
    h, w = dp.shape[:2]
    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d

def pose_draw():
    cx7,cy7,cx9,cy9,cx5,cy5,l,r=0,0,0,0,0,0,0,0
    s1,s2,s3,s4=0,0,0,0
    global ax,ay,az,bx,by,bz,frame2
    #for num in [7,9]: #[7,9] left, [8,10] right
    n1,n2,n3=12,14,16
    cx7, cy7 = get_pose_target(pose,n2)
    
    cx9, cy9 = get_pose_target(pose,n3)
    
    cx5, cy5 = get_pose_target(pose,n1)
    if cx7==-1 and cx9!=-1:
        s1,s2,s3,s4=cx5,cy5,cx9,cy9
        cv2.circle(frame2, (cx5,cy5), 5, (0, 255, 0), -1)
        ax,ay,az = get_real_xyz(depth2,cx5, cy5)
        _,_,l=get_real_xyz(depth2,cx5,cy5)
        cv2.circle(frame2, (cx9,cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(depth2,cx9, cy9)
        _,_,r=get_real_xyz(depth2,cx9,cy9)
    elif cx7 !=-1 and cx9 ==-1:
        s1,s2,s3,s4=cx5,cy5,cx7,cy7
        cv2.circle(frame2, (cx5,cy5), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(depth2,cx5, cy5)
        _,_,l=get_real_xyz(depth2,cx5,cy5)
        cv2.circle(frame2, (cx7,cy7), 5, (0, 255, 0), -1)
        bx,by,bz = get_real_xyz(depth2,cx7, cy7)
        _,_,r=get_real_xyz(depth2,cx7,cy7)
    elif cx7 ==-1 and cx9 == -1:
        pass
        #continue
    else:
        s1,s2,s3,s4=cx7,cy7,cx9,cy9
        cv2.circle(frame2, (cx7,cy7), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(depth2,cx7, cy7)
        _,_,l=get_real_xyz(depth2,cx7,cy7)
        cv2.circle(frame2, (cx9,cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(depth2,cx9, cy9)
        _,_,r=get_real_xyz(depth2,cx9,cy9)
    
    return ay,by
    #cv2.putText(frame, str(int(l)), (s1 + 5, s2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    #cv2.putText(frame, str(int(r)), (s3 + 5, s4 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
def get_pose_target(pose,num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])
    
    if len(p) == 0: return -1, -1
    return int(p[0][0]),int(p[0][1])
def say(a):
    global publisher_speaker
    publisher_speaker.publish(a)
if __name__ == "__main__": 
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    
    frame2 = None
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image2)

    depth2 = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth2)
    
    frame1 = None
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image1)

    depth1= None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth1)
    
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    print("load")
    dnn_yolo = Yolov8("yolov8n", device_name="GPU")
    print("yolo")
    say("start the program")
    net_pose = HumanPoseEstimation(device_name="GPU")
    step="fall"
    fcnt=0
    step2="dead"
    while not rospy.is_shutdown():
        rospy.Rate(10).sleep()
        
        if frame1 is None: 
            print("frame1")
            continue
        if frame2 is None: 
            print("frame2")
            continue
        if depth1 is None: 
            print("depth1")
            continue
        if depth2 is None: 
            print("depth2")
            continue
        if step=="fall":
            if f_cnt>=3: step="get"
            detections = dnn_yolo.forward(frame2)[0]["det"]
            
            show=frame2
            showd=depth2
            for i, detection in enumerate(detections):
                #time.sleep(0.001)
                fall,ikun=0,0
                x1, y1, x2, y2, score, class_id = map(int, detection)
                score = detection[4]
                if class_id != 0:
                    continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                px,py,pz=get_real_xyz(depth2, cx, cy)
                if pz<=2000:
                    pose=None
                    poses = net_pose.forward(frame2)
                    t_pose=None
                    points=[]
                    for i, pose in enumerate(poses):
                        point = []
                        for j, (x,y,preds) in enumerate(pose): #x: ipex 坐標 y: ipex 坐標 preds: 准度
                            if preds <= 0: continue
                            x,y = map(int,[x,y])
                            _,_,td=get_real_xyz(depth2,x, y)
                            if td>=2000: continue
                            if j in [14,16]:
                                point.append(j)
                        if len(point) == 2:
                            t_pose = poses[i]
                            break
                        #print(point)
                    TTT=0
                    E=0
                    s_c=[]
                    
                    s_d=[]
                    ggg=0
                    flag=None
                    if t_pose is not None:
                        cy,dy=pose_draw()
                        if abs(cy-dy)<=50: ikun=99
                        
                    #print(x2,x1,y1,y2)
                    w=x2-x1
                    h=y2-y1
                    print("w: ",w,"h: ", h)
                    w,h=w,h
                    if cy<=160:
                        fall+=1
                        print("cy")
                    if h<w:
                        fall+=1
                        print("h<w")
                    if fall>=1 and ikun==99:
                        cv2.rectangle(frame2, (x1, y1), (x2, y2), (255, 255, 0), 5)
                        fcnt+=1
                        continue
                    else:
                        cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 0,255), 1)
                        continue

        if step=="get":
            bottle=[]
            detections = dnn_yolo.forward(frame2)[0]["det"]
            al=[]
            ind=0
            for i, detection in enumerate(detections):
                x1, y1, x2, y2, score, class_id = map(int, detection)
                if class_id != 39: continue
                al.append([x1, y1, x2, y2, score, class_id])
            bb=sorted(al, key=(lambda x:x[0]))
            #print(bb)
            for i in bb:
                #print(i)
                x1, y1, x2, y2, _, _ = i
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 255), 4)
                cv2.putText(frame2, str(int(ind)), (cx,cy+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                ind+=1
                px,py,pz=get_real_xyz(depth2,cx, cy)
                cnt=get_distance(px,py,pz,ax,ay,az,bx,by,bz)
                cv2.circle(frame2, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(frame2, str(int(pz)), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            if step2=="dead":
                outframe=frame2.copy()
                t_pose=None
                points=[]
                poses = net_pose.forward(outframe)
                
                for i, pose in enumerate(poses):
                    point = []
                    for j, (x,y,preds) in enumerate(pose): #x: ipex 坐標 y: ipex 坐標 preds: 准度
                        if preds <= 0: continue
                        x,y = map(int,[x,y])
                        _,_,td=get_real_xyz(depth2,x, y)
                        if td>=2000: continue
                        if j in [8,10]:
                            point.append(j)
                    if len(point) == 2:
                        t_pose = poses[i]
                        break
                    #print(point)
                
                TTT=0
                E=0
                s_c=[]
                
                s_d=[]
                ggg=0
                flag=None
                if t_pose is not None:
                    ax, ay, az, bx, by, bz = pose_draw()
                    for i, detection in enumerate(bb):
                        #print(detection)
                        x1, y1, x2, y2, score, class_id = map(int, detection)
                        score = detection[4]
                        #print(id)
                        if(class_id == 39):
                            ggg=1
                            bottle.append(detection)
                            E+=1
                            cx1 = (x2 - x1) // 2 + x1
                            cy1 = (y2 - y1) // 2 + y1
                            
                            
                            px,py,pz=get_real_xyz(depth2, cx1, cy1)
                            cnt=get_distance(px,py,pz,ax,ay,az,bx,by,bz)
                            
                            cnt=int(cnt)
                            if cnt!=0 and cnt<=600: cnt=int(cnt)
                            else: cnt=9999
                            s_c.append(cnt)
                            s_d.append(pz)
                            
                if ggg==0: s_c=[9999]
                TTT=min(s_c)
                E=s_c.index(TTT)
                for i, detection in enumerate(bottle):
                    #print("1")
                    x1, y1, x2, y2, score, class_id = map(int, detection)
                    if(class_id == 39):
                        if i == E and E!=9999 and TTT <=700:
                            cx1 = (x2 - x1) // 2 + x1
                            cy1 = (y2 - y1) // 2 + y1
                            cv2.putText(outframe, str(int(TTT)//10), (x1 + 5, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 0, 255), 2)
                            cv2.rectangle(outframe, (x1, y1), (x2, y2), (0, 0, 255), 5)
                            if i==0: b1+=1
                            if i==1: b2+=1
                            if i==2: b3+=1
                            
                            break
                                    
                        else:
                            v=s_c[i]
                            cv2.putText(outframe, str(int(v)), (x1+5, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if b1==max(b1,b2,b3): mark=0
                if b2==max(b1,b2,b3): mark=1
                if b3==max(b1,b2,b3): mark=2
                if b1 >=10 or b2>=10 or b3>=10: 
                    step2="get"
                    gg=bb
                print("b1: %d b2: %d b3: %d" % (b1, b2, b3))
            if step2=="get":
                if len(bb)!=3: continue
                print(bb)
                h,w,c = outframe.shape
                x1, y1, x2, y2, score, class_id = map(int, bb[mark])
                cx2 = (x2 - x1) // 2 + x1
                cy2 = (y2 - y1) // 2 + y1
                msg=Twist()
                #move1(cx2,cy2, msg)
                rx, ry, rz = get_real_xyz(depth2,cx2, cy2)
                if(rz==0): continue
                angle = np.arctan2(rx, rz)
                print(angle)
                msg.angular.z=-angle
                _cmd_vel.publish(msg)
                
                
                cx, cy = w // 2, h // 2
                for i in range(cy + 1, h):
                    if depth2[cy][cx] == 0 or 0 < depth2[i][cx] < depth2[cy][cx]:
                        cy = i 
                _,_,d = get_real_xyz(depth2,cx,cy)
                while d > 0 or abs(e) >= 1:
                    _,_,d1 = get_real_xyz(depth2,cx,cy)
                    e = d1 - 550 #number is he last distance
                    if abs(e)<=1:
                        break
                    v = 0.001 * e
                    if v > 0:
                        v = min(v, 0.2)
                    if v < 0:
                        v = max(v, -0.2)
                    print(d1, e, v)
                    move(v, 0)
                step="grap"
                step2="no"
        if step=="grap":
            t=3.0
            time.sleep(3.0)
            joint1, joint2, joint3, joint4 = 0.000, 1.0, -0.5,-0.6
            set_joints(joint1, joint2, joint3, joint4, t)
            time.sleep(t)
            close_gripper(t)
            step="turnh"
        if step=="turnh":
            angle_rad = math.atan(60/1)
            angle_deg = math.degrees(angle_rad)
            msg.angular.z=-angle
            _cmd_vel.publish(msg)
            msg.angular.z=0
        if step="givehim":
            detections = dnn_yolo.forward(frame2)[0]["det"]
            dc=999999
            mcx,mcy=0,0
            for i, detection in enumerate(detections):
                x1, y1, x2, y2, score, class_id = map(int, detection)
                if class_id != 0: continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                _,_,gd=get_real_xyz(depth2,cx,cy)
                if gd<dc:
                    dc=gd
                    mcx,mcy=cx,cy
            rx, ry, rz = get_real_xyz(depth2,mcx, mcy)
            if(rz==0 and mx!=0): continue
            angle = np.arctan2(rx, rz)
            print(angle)
            msg.angular.z=-angle
            _cmd_vel.publish(msg)
            step == "put"
        if step=="put":
            #go front
            joint1, joint2, joint3, joint4 = 0.000, 0.0, -0.5,1.0
            set_joints(joint1, joint2, joint3, joint4, t)
            time.sleep(t)
            open_gripper(t)
            time.sleep(2)
            joint1, joint2, joint3, joint4 = 0.000, -1.0, 0.3,0.70
            set_joints(joint1, joint2, joint3, joint4, t)
            
            time.sleep(2.5)
            step="back"
        if step == "back":
            chassis.move_to(-6.82,-6.37,0.00322)
            #checking
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
                if "robot" in s and "stop" in s:
                    break
            break
        cv2.imshow("image", frame2)   
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
        
