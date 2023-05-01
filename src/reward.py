#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64


if __name__ == '__main__':
    rospy.init_node('reward')
    pub = rospy.Publisher('reward', Float64, queue_size=10)
    msg = Float64()

    while True:

        msg.data = float(input('reward = '))
        pub.publish(msg)
        msg.data = float(input('terminal = '))
        pub.publish(msg)

