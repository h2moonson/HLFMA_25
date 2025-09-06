#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray

pub = None

def joy_callback(msg):
    data = Float32MultiArray()
    speed = msg.axes[4]  # -1.0 ~ 1.0
    steer = msg.axes[0]  # -1.0 ~ 1.0
    button0 = float(msg.buttons[0]) #A: Autonomous
    button1 = float(msg.buttons[1]) #B: Manual 
    button2 = float(msg.buttons[2]) #X: e-stop
		
    data.data = [speed, steer, button0, button1, button2]
    pub.publish(data)

def main():
    global pub
    rospy.init_node('xbox_to_array')
    pub = rospy.Publisher('xbox_cmd', Float32MultiArray, queue_size=10)
    rospy.Subscriber('/joy', Joy, joy_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
