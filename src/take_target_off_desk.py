#!/usr/bin/env python

from time import time
import rospy
from moveit_commander import MoveGroupCommander
from actionlib_msgs.msg import GoalStatusArray
from vgn.utils.transform import Transform, Rotation
from scipy.spatial.transform import Rotation as R
from vgn.utils.panda_control import PandaCommander
import numpy as np
from tf2_ros import TransformListener, Buffer

def take_target_off_desk():
    rospy.init_node('take_target_off_desk')
    tfBuffer = Buffer()
    listener = TransformListener(tfBuffer)  # can not be delete
    rospy.wait_for_message('move_group/status', GoalStatusArray)
    trans = tfBuffer.lookup_transform('world', 'panda_NE', rospy.Time(), timeout=rospy.Duration(100))
    commander = MoveGroupCommander('panda_arm')
    pc = PandaCommander()
    pc.grasp(width=0.0, force=20.0)
    ###############################################################################
    action = np.array(
            [trans.transform.translation.x, trans.transform.translation.y, 0.062,
             trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z,
             trans.transform.rotation.w])
    T_tool0_tcp = Transform.from_dict({"rotation": [0, 0, -0.382, 0.924], "translation": [0, 0, 0.103399999]})
    T_tcp_tool0 = T_tool0_tcp.inverse()
    r = R.from_quat(np.array(action[3:7]))
    T = Transform(r, np.array(action[:3]))
    pc.goto_pose(T*T_tcp_tool0, velocity_scaling=0.03)
    ####################################'ready'#####################################
    pc.move_gripper(0.08)
    pc.goto_joints([0.05331221075800427, -0.27569424368626394, 0.11116034932931264, -1.614720256085981,
                    0.07154469995857352, 1.2394388119808621, 0.9599727860593279], 0.03, 0.2)

    commander.set_named_target('ready')
    commander.set_max_velocity_scaling_factor(0.1)
    commander.go()
    ################################################################################
    # pc.grasp(width=0.0, force=20.0)
    # rospy.sleep(0.3)
    # pc.move_gripper(0.08)



if __name__ == '__main__':

    take_target_off_desk()
