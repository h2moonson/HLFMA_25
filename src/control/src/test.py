#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import time

class Test:
    def __init__(self):
        rospy.init_node('test_node')
        self.pub = rospy.Publisher('teleop_cmd_vel', Twist, queue_size=1)

    def publish_constantly(self, data, duration_sec, rate_hz=30):
        msg = Twist()
        msg.linear.x = data
        rate = rospy.Rate(rate_hz)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if (now - start_time).to_sec() >= duration_sec:
                break
            self.pub.publish(msg)
            rate.sleep()

if __name__ == '__main__':
    test = Test()
    while not rospy.is_shutdown():
        test.publish_constantly(50, duration_sec=2, rate_hz=30)
        test.publish_constantly(0, duration_sec=2, rate_hz=30)
