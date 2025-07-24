import rospy
from nav_msgs.msg import Odometry
import sys
import threading

triger = 0
def publish_position():
    global target_x, target_y, target_z, yaw, triger
    msg = Odometry()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "world"
    msg.pose.pose.position.x = target_x
    msg.pose.pose.position.y = target_y
    msg.pose.pose.position.z = target_z
    msg.pose.pose.orientation.w = 0
    pos_pub.publish(msg)
    triger += 1
    if triger >= 30:
        rospy.loginfo(f'Publishing: x={target_x}, y={target_y}, z={target_z}')
        triger = 0

def listen_input():
    global target_x,target_z, target_y
    while not rospy.is_shutdown():
        user_input = sys.stdin.readline().strip()
        if user_input == '1':
            target_x = 50
            target_y = -4
            target_z = 1.3
            print("可以返航了")
        else:
            target_x = 0.0

if __name__ == "__main__":
    rospy.init_node('dog_pos_publisher', anonymous=True)
    pos_pub = rospy.Publisher('/dog_pos', Odometry, queue_size=10)
    rate = rospy.Rate(20)  # 20Hz
    
    target_x, target_y, target_z = 0.0, 0.0, 0.5
    yaw = 0.0
    
    thread = threading.Thread(target=listen_input, daemon=True)
    thread.start()
    
    while not rospy.is_shutdown():
        publish_position()
        rate.sleep()