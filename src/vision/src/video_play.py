#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospkg
import os

def main():
    rospy.init_node('video_publisher_node')
    pub = rospy.Publisher('/usb_cam/image_raw', Image, queue_size=10)
    bridge = CvBridge()
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('vision')
    video_path = os.path.join(pkg_path, 'test.mp4')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        rospy.logerr("Cannot open video: {}".format(video_path))
        return

    rate = rospy.Rate(30)  # 30fps
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.loginfo("Video ended or cannot fetch the frame.")
            break

        try:
            msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            pub.publish(msg)
        except Exception as e:
            rospy.logerr("cv_bridge error: {}".format(e))

        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
