def move_robot(forward_speed, turn_speed):
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)

def move_distance(distance):
    distance = distance / 1000.0
    forward_speed = 0.2
    forward_time = abs(distance) / forward_speed
    move_robot(forward_speed, 0)
    rospy.sleep(forward_time)
    move_robot(0, 0)

move_distance(1000)
