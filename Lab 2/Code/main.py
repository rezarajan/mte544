#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
import tf

import math
from math import sin, cos, pi,tan, atan2
import numpy as np

from localmap import localmap
from run_ekf import run_ekf

pose=[0.0,0.0,0.0]
odom_pose=[0.0,0.0,0.0]
scan_time = None
pos_time = None
odom_time = None
ekf_time = None
marker_time = None
marker_time_noise = None
marker_time_ekf = None
marker_time_ekf_ellipse = None
marker = Marker()
marker_noise = Marker()
marker_ekf = Marker()
count = 0

state_array = np.zeros((3,1))
P = np.identity(3)

#***********************************************************************
def handle_robot_pose(parent, child, pose):
    br = tf.TransformBroadcaster()
    br.sendTransform((pose[0], pose[1], 0), tf.transformations.quaternion_from_euler(0, 0, pose[2]), rospy.Time.now(), child, parent)

#***********************************************************************
def positionCb(msg):
    global pose
    global pos_time
    x=msg.pose.pose.position.x
    y=msg.pose.pose.position.y
    q0 = msg.pose.pose.orientation.w
    q1 = msg.pose.pose.orientation.x
    q2 = msg.pose.pose.orientation.y
    q3 = msg.pose.pose.orientation.z
    theta=atan2(2*(q0*q3+q1*q2),1-2*(q2*q2+q3*q3)) # heading angle
    # Correcting for RViz coordinate system: 0-180, then -180 to 0 instead of 0-360
    if theta > 0:
        pass
    elif theta < 0:
        theta += 2*math.pi
    pose=[x,y,theta]

    # show_marker(msg)

    pos_time = rospy.Time.now()

#***********************************************************************
def odometryCb(msg):
    global odom_pose
    global odom_time, ekf_time
    x=msg.pose.pose.position.x
    y=msg.pose.pose.position.y
    q0 = msg.pose.pose.orientation.w
    q1 = msg.pose.pose.orientation.x
    q2 = msg.pose.pose.orientation.y
    q3 = msg.pose.pose.orientation.z
    theta=atan2(2*(q0*q3+q1*q2),1-2*(q2*q2+q3*q3)) # heading angle
    # Correcting for RViz coordinate system: 0-180, then -180 to 0 instead of 0-360
    if theta > 0:
        pass
    elif theta < 0:
        theta += 2*math.pi
    odom_pose=[x,y,theta]
    
    odom_time = rospy.Time.now()
    ekf_time = rospy.Time.now()

    show_marker()
    show_marker_noise(msg)

    handle_robot_pose("map", "odom", pose)
    handle_robot_pose("map", "base_link", pose)

#*********************************************************************** 
def scanCb(msg):

    global scan_time
    global pos_time

    scan_time = rospy.Time.now()

    if scan_time is not None and pos_time is not None and rospy.Duration(nsecs=0) <= (scan_time-pos_time) <= rospy.Duration(nsecs=1000000):
        scandata=msg.ranges
        angle_min=msg.angle_min
        angle_max=msg.angle_max
        angle_increment=msg.angle_increment
        range_min=msg.range_min
        range_max=msg.range_max 
        m.updatemap(scandata,angle_min,angle_max,angle_increment,range_min,range_max,pose)
        # handle_robot_pose("map", "indoor_pos", pose)
        # handle_robot_pose("map", "odom", pose)
        # handle_robot_pose("map", "base_link", pose)


#***********************************************************************    
def mappublisher(m,height, width, resolution, morigin):
    msg = OccupancyGrid()
    msg.header.frame_id='map'
    msg.info.resolution = resolution
    msg.info.width      = math.ceil(width/resolution)
    msg.info.height     = math.ceil(height/resolution)
    msg.info.origin.position.x=-morigin[0]
    msg.info.origin.position.y=-morigin[1]
    msg.data=m  
    mappub.publish(msg)


def show_marker():

    global marker, pose
    global marker_time
    POINTS_MAX = 100


    if marker_time is None or (rospy.Time.now()-marker_time) >= rospy.Duration(secs=4):


        marker.header.frame_id = "/map"
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD

        # marker scale
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03

        # marker color
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        # marker orientaiton
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # marker position
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0

        # marker line points
        line_point = Point()
        line_point.x = pose[0]
        line_point.y = pose[1]
        line_point.z = pose[2]

        # Add the new marker to the points array, removing the oldest
        # marker from it when necessary
        if(count > POINTS_MAX):
            marker.points.pop(0)

        marker.points.append(line_point)    

        markerpub = rospy.Publisher("/visualization_truth", Marker, queue_size=100)
        markerpub.publish(marker)

        marker_time = rospy.Time.now()

def show_marker_noise(msg):

    global marker_noise, pose
    global marker_time_noise, ekf_time, odom_time
    global state_array, P
    global pose
    POINTS_MAX = 100

    pose_noise = np.add(np.array([pose]), [np.random.normal(0, 0.1), np.random.normal(0, 0.1), np.random.normal(0, 0.1)])


    if ekf_time is not None and odom_time is not None and pos_time is not None:
        timestep = (odom_time-ekf_time).secs
        sensor = pose_noise.reshape(3,-1)
        
        u = np.array([[msg.twist.twist.linear.x, msg.twist.twist.angular.z]]).reshape(2,-1)
        ekf = run_ekf(state_array, u, sensor, P, timestep)
        x_pred, P, X0, Y0, a, b, angle = ekf.run()
        state_array = np.append(state_array, x_pred.reshape(3,-1), axis=1)


        if marker_time_noise is None or (rospy.Time.now()-marker_time_noise) >= rospy.Duration(secs=4):


            marker_noise.header.frame_id = "/map"
            marker_noise.type = marker.LINE_STRIP
            marker_noise.action = marker.ADD

            # marker scale
            marker_noise.scale.x = 0.03
            marker_noise.scale.y = 0.03
            marker_noise.scale.z = 0.03

            # marker color
            marker_noise.color.a = 1.0
            marker_noise.color.r = 1.0
            marker_noise.color.g = 0.0
            marker_noise.color.b = 0.0

            # marker orientaiton
            marker_noise.pose.orientation.x = 0.0
            marker_noise.pose.orientation.y = 0.0
            marker_noise.pose.orientation.z = 0.0
            marker_noise.pose.orientation.w = 1.0

            # marker position
            marker_noise.pose.position.x = 0.0
            marker_noise.pose.position.y = 0.0
            marker_noise.pose.position.z = 0.0

            # marker line points
            line_point = Point()
            line_point.x = pose_noise[0][0]
            line_point.y = pose_noise[0][1]
            line_point.z = pose_noise[0][2]

            # Add the new marker to the points array, removing the oldest
            # marker from it when necessary
            if(count > POINTS_MAX):
                marker_noise.points.pop(0)

            marker_noise.points.append(line_point)    

            markerpub = rospy.Publisher("/visualization_noise", Marker, queue_size=100)
            markerpub.publish(marker_noise)

            marker_time_noise = rospy.Time.now()
            print("Real: x: {} | y: {}".format(pose[0],pose[1]))
            print("Noise: x: {} | y: {}".format(pose_noise[0][0],pose_noise[0][1]))

            show_marker_ekf()
            show_marker_ekf_ellipse(X0, Y0, a, b, angle)

def show_marker_ekf():

    global marker_ekf, state_array
    global marker_time_ekf
    POINTS_MAX = 100

    ekf_pose = state_array[:,-1]


    if marker_time_ekf is None or (rospy.Time.now()-marker_time_ekf) >= rospy.Duration(secs=4):


        marker_ekf.header.frame_id = "/map"
        marker_ekf.type = marker.LINE_STRIP
        marker_ekf.action = marker.ADD

        # marker scale
        marker_ekf.scale.x = 0.03
        marker_ekf.scale.y = 0.03
        marker_ekf.scale.z = 0.03

        # marker color
        marker_ekf.color.a = 1.0
        marker_ekf.color.r = 1.0
        marker_ekf.color.g = 0.0
        marker_ekf.color.b = 1.0

        # marker orientaiton
        marker_ekf.pose.orientation.x = 0.0
        marker_ekf.pose.orientation.y = 0.0
        marker_ekf.pose.orientation.z = 0.0
        marker_ekf.pose.orientation.w = 1.0

        # marker position
        marker_ekf.pose.position.x = 0.0
        marker_ekf.pose.position.y = 0.0
        marker_ekf.pose.position.z = 0.0

        # marker line points
        line_point = Point()
        line_point.x = ekf_pose[0]
        line_point.y = ekf_pose[1]
        line_point.z = ekf_pose[2]

        # Add the new marker to the points array, removing the oldest
        # marker from it when necessary
        if(count > POINTS_MAX):
            marker_ekf.points.pop(0)

        marker_ekf.points.append(line_point)    

        markerpub = rospy.Publisher("/visualization_ekf", Marker, queue_size=100)
        markerpub.publish(marker_ekf)

        marker_time_ekf = rospy.Time.now()
        print("EKF Pose: x: {} | y: {}".format(ekf_pose[0],ekf_pose[1]))


def show_marker_ekf_ellipse(X0, Y0, a, b, angle):

    global marker_time_ekf_ellipse, ekf_time, odom_time
    global state_array, P
    global pose

    if marker_time_ekf_ellipse is None or (rospy.Time.now()-marker_time_ekf_ellipse) >= rospy.Duration(secs=4):

        marker = Marker()
        marker.header.frame_id = "/map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = a
        marker.scale.y = b
        marker.scale.z = 0.0
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        # marker position
        marker.pose.position.x = X0
        marker.pose.position.y = Y0
        marker.pose.position.z = 0.0
        
        q = tf.transformations.quaternion_from_euler(0, 0, angle)
        # marker orientaiton
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]

        markerpub = rospy.Publisher("/visualization_ekf_ellispe", Marker, queue_size=1)
        markerpub.publish(marker)

        marker_time_ekf_ellipse = rospy.Time.now()

        # print("x: {} | y: {}".format(a,b))
        # print(q)


if __name__ == "__main__":

    rospy.init_node('main', anonymous=True) #make node 
    rospy.Subscriber('/odom',Odometry,odometryCb)
    rospy.Subscriber('/indoor_pos',PoseWithCovarianceStamped,positionCb)
    rospy.Subscriber("/scan", LaserScan, scanCb, queue_size=1)
    mappub= rospy.Publisher('/map', OccupancyGrid, queue_size=1)

    rate = rospy.Rate(100) # 100hz   

    height, width, resolution=30,30,0.05
    morigin=[width/2.0,height/2.0]
    m=localmap(height, width, resolution, morigin)
    

    while not rospy.is_shutdown():
        mappublisher(m.localmap, height, width, resolution, morigin)
        rate.sleep()
