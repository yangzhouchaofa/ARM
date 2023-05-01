#!/usr/bin/env python

from time import time
import rospy
from moveit_commander import MoveGroupCommander
from actionlib_msgs.msg import GoalStatusArray
from vgn.utils.transform import Transform, Rotation
from scipy.spatial.transform import Rotation as R
from vgn.utils.panda_control import PandaCommander
import numpy as np


def move_to_start():
    rospy.init_node('move_to_start')
    rospy.wait_for_message('move_group/status', GoalStatusArray)
    commander = MoveGroupCommander('panda_arm')
    pc = PandaCommander()
    pc.move_gripper(0.08)
    ###############################################################################
    # action = [ 0.43097669, -0.08807754 , 0.12882341, -0.99587201, -0.08957327,  0.01429179, -0.00238249]
    # T_tool0_tcp = Transform.from_dict({"rotation": [0, 0, -0.382, 0.924], "translation": [0, 0, 0.103399999]})
    # T_tcp_tool0 = T_tool0_tcp.inverse()
    # r = R.from_quat(np.array(action[3:7]))
    # T = Transform(r, np.array(action[:3]))
    # pc.goto_pose(T*T_tcp_tool0, velocity_scaling=0.08)
    pc.goto_joints([0.05331221075800427, -0.27569424368626394, 0.11116034932931264, -1.614720256085981,
                    0.07154469995857352, 1.2394388119808621, 0.9599727860593279], 0.1, 0.2)
    ####################################'ready'#####################################
    # commander.set_named_target('ready')
    # commander.set_max_velocity_scaling_factor(0.1)
    # commander.go()
    ################################################################################
    # pc.grasp(width=0.0, force=20.0)
    # rospy.sleep(0.3)
    # pc.move_gripper(0.08)



if __name__ == '__main__':

    move_to_start()
