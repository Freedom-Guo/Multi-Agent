import csv
import numpy as np
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import *
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from gazebo_msgs.msg import ModelStates
import rospy
import tf
import math

##----------------------------------全局参数列表------------------------------------------##
agent_num = 3
evader_num = 1
pursuer_num = 2
death_num = 0

Exit_Pos = [-100., 100., -100., 100.]

roll = []
pitch = []
yaw = []
linear_x = []
angular_z = []
pos_x = np.array([0., -5., -7., -5., 4.])
pos_y = np.array([0., 0., -5., 5., -3.])

flag_capture = False
death = []

for i in range(agent_num):
    roll.append(0.)
    pitch.append(0.)
    yaw.append(0.)
    linear_x.append(0.)
    angular_z.append(0.)


##---------------------------------计算三角形的外心----------------------------------------##
## A B C为三角形的三个顶点
## k为斜率值 ab和bc为对应边的角度
## (x,y)为外接圆圆心 r为半径
def get_outer_circle(A, B, C):
    # 顶点的坐标
    xa, ya = A[0], A[1]
    xb, yb = B[0], B[1]
    xc, yc = C[0], C[1]

    # 两条边的中点
    xab, yab = (xa + xb) / 2.0, (ya + yb) / 2.0
    xbc, ybc = (xb + xc) / 2.0, (yb + yc) / 2.0

    # 两条边的斜率
    if xb != xa:
        kab = (yb - ya) / (xb - xa)
    else:
        kab = None

    if xc != xb:
        kbc = (yc - yb) / (xc - xb)
    else:
        kbc = None

    if kab is not None:
        ab = np.arctan(kab)
    else:
        ab = np.pi / 2

    if kbc is not None:
        bc = np.arctan(kbc)
    else:
        bc = np.pi / 2

    # 两条边的中垂线
    if ab == 0:
        kabm = None
        b1 = 0
        x = xab
    else:
        kabm = np.tan(ab + np.pi / 2)
        b1 = yab * 1.0 - xab * kabm * 1.0

    if bc == 0:
        kbcm = None
        b2 = 0
        x = xbc
    else:
        kbcm = np.tan(bc + np.pi / 2)
        b2 = ybc * 1.0 - xbc * kbcm * 1.0

    if kabm is not None and kbcm is not None:
        x = (b2 - b1) * 1.0 / (kabm - kbcm)

    if kabm is not None:
        y = kabm * x * 1.0 + b1 * 1.0
    else:
        y = kbcm * x * 1.0 + b2 * 1.0

    r = np.sqrt((x - xa) ** 2 + (y - ya) ** 2)
    return x, y, r


##---------------------------------获取直线和边界的交点------------------------------------##
## a b c为直线解析式ax+by+c=0中的参数
## bound为边界的限制，例如矩阵：bound[0,1,2,3]分别表示x的最小最大值以及y的最小最大值
## (x1, y1)和(x2, y2)为交点的坐标
## flag表示寻找到交点的数量
def get_intersect_point(a, b, c, bound):
    flag = 0
    x1 = y1 = x2 = y2 = 0

    if b == 0:
        # 斜率不存在
        x1 = x2 = -c / a
        y1 = bound[2]
        y2 = bound[3]
    else:
        # 斜率存在
        if bound[3] >= (-c - a * bound[0]) / b >= bound[2]:
            # 线和x=bound[0]存在符合要求的交点
            if flag == 0:
                x1 = bound[0]
                y1 = (-c - a * bound[0]) / b
                flag = 1
            else:
                x2 = bound[0]
                y2 = (-c - a * bound[0]) / b
                flag = 2

        if bound[3] >= (-c - a * bound[1]) / b >= bound[2]:
            # 线和x=bound[1]存在符合要求的交点
            if flag == 0:
                x1 = bound[1]
                y1 = (-c - a * bound[1]) / b
                flag = 1
            else:
                # 找到过符合要求的交点
                x2 = bound[1]
                y2 = (-c - a * bound[1]) / b
                flag = 2

        if bound[1] >= (-c - b * bound[2]) / a >= bound[0]:
            # 线和y=bound[2]存在符合要求的交点
            if flag == 0:
                y1 = bound[2]
                x1 = (-c - b * bound[2]) / a
                flag = 1
            else:
                y2 = bound[2]
                x2 = (-c - b * bound[2]) / a
                flag = 2

        if bound[1] >= (-c - b * bound[3]) / a >= bound[0]:
            # 线和y=bound[3]存在符合要求的交点
            if flag == 0:
                y1 = bound[3]
                x1 = (-c - b * bound[3]) / a
                flag = 1
            else:
                y2 = bound[3]
                x2 = (-c - b * bound[3]) / a
                flag = 2
        if flag == 1:
            # 只存在一个交点
            x2 = x1
            y2 = y1

    # print("线和边界的交点")
    # print([x1, y1], [x2, y2])
    return flag, x1, y1, x2, y2


##-----------------------------获取Voronoit图中需要连接的两点-------------------------------##
## A B为需要连接的两点，因为需要考虑其中点在边界以外，这个需要截取线段在边界内部的部分，如果都在外面则舍弃
## flag 1表示A、B两点不是都在边界以外 0表示A、B两点都在边界以外
## C表示截取线段中位于边界上的端点
def intersect(A, B, bound):
    C = [0, 0]
    if bound[0] <= A[0] <= bound[1] and bound[2] <= A[1] <= bound[3]:
        # A点在区域内部
        # print("A点在区域内部")
        if bound[0] <= B[0] <= bound[1] and bound[2] <= B[1] <= bound[3]:
            # B点在区域内
            flag = 1
            return A[0], A[1], B[0], B[1], flag
        else:
            # B点不在区域内
            flag = 1
            if A[0] == B[0]:
                # AB的斜率不存在
                if B[1] > bound[3]:
                    x = A[0]
                    y = bound[3]
                else:
                    x = A[0]
                    y = bound[2]
                C[0] = x
                C[1] = y
            else:
                # AB的斜率存在
                a = A[1] - B[1]
                b = B[0] - A[0]
                c = B[1] * A[0] - A[1] * B[0]
                num, x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
                if min(A[0], B[0]) <= x1 <= max(A[0], B[0]) and min(A[1], B[1]) <= y1 <= max(A[1], B[1]):
                    C[0] = x1
                    C[1] = y1
                else:
                    C[0] = x2
                    C[1] = y2
            return A[0], A[1], C[0], C[1], flag
    else:
        # A点不在区域内部
        if bound[0] <= B[0] <= bound[1] and bound[2] <= B[1] <= bound[3]:
            # B点在区域内
            flag = 1
            if A[0] == B[0]:
                # AB的斜率不存在
                if A[1] > bound[3]:
                    x = B[0]
                    y = bound[3]
                else:
                    x = B[0]
                    y = bound[2]
                C = [x, y]
            else:
                # AB的斜率存在
                a = A[1] - B[1]
                b = B[0] - A[0]
                c = B[1] * A[0] - A[1] * B[0]
                num, x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
                if min(A[0], B[0]) <= x1 <= max(A[0], B[0]) and min(A[1], B[1]) <= y1 <= max(A[1], B[1]):
                    C[0] = x1
                    C[1] = y1
                else:
                    C[0] = x2
                    C[1] = y2
            return B[0], B[1], C[0], C[1], flag
        else:
            flag = 0
            if A[0] == B[0]:
                return A[0], A[1], B[0], B[1], flag
            else:
                a = A[1] - B[1]
                b = B[0] - A[0]
                c = B[1] * A[0] - A[1] * B[0]
                num, x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
                if num > 0:
                    return x1, y1, x2, y2, flag
                else:
                    return A[0], A[1], B[0], B[1], flag


##----------------------------------判断两点是否位于直线异侧--------------------------------##
## p1和p2为直线上的两点，p3和p4是需要进行判断的两点
## a b c为直线p1p2解析式ax+by+c=0中的参数
def IsIntersec(p1, p2, p3, p4):
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p2[0] * p1[1] - p1[0] * p2[1]
    if (a * p3[0] + b * p3[1] + c) * (a * p4[0] + b * p4[1] + c) <= 0:
        return 1
    else:
        return 0


##-------------------------------获取中垂线和边界符合要求的交点------------------------------##
## A B为需要求中垂线的的线段的两个端点 C为三角形中除了A和B以外剩下的顶点 D为符合要求的交点
## a b c为中垂线解析式中的参数
## (x1, y1) (X2,y2)为中垂线和边界的两个交点的坐标
def midline(A, B, C, bound):
    a = 2 * (B[0] - A[0])
    b = 2 * (B[1] - A[1])
    c = A[0] ** 2 - B[0] ** 2 + A[1] ** 2 - B[1] ** 2
    num, x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
    D = [x1, y1]
    if IsIntersec(A, B, C, D):
        D = [x1, y1]
    else:
        D = [x2, y2]
    return D


##-----------------------------------获取li的两个端点-------------------------------------##
def get_l_point(A, B, C, D):
    # A-交点1 B-交点2 C-evader D-pursuer
    x1 = (C[0] + D[0]) / 2.
    y1 = (C[1] + D[1]) / 2.
    v1 = np.array([D[0] - A[0], D[1] - A[1]])
    v2 = np.array([D[0] - B[0], D[1] - B[1]])
    v3 = np.array([D[0] - C[0], D[1] - C[1]])
    if np.cross(v1, v3) > 0 and np.cross(v2, v3) > 0:
        if ((A[0] - x1) ** 2 + (A[1] - y1) ** 2) > ((B[0] - x1) ** 2 + (B[1] - y1) ** 2):
            x2 = A[0]
            y2 = A[1]
        else:
            x2 = B[0]
            y2 = B[1]
    elif np.cross(v1, v3) > 0:
        x2 = A[0]
        y2 = A[1]
    else:
        x2 = B[0]
        y2 = B[1]
    return x1, y1, x2, y2


def is_tri(tri):
    flag_p = False
    flag_e = False
    global death, evader_num, agent_num
    for i in range(evader_num - len(death)):
        if tri[0] == i or tri[1] == i:
            flag_e = True
            break
        else:
            flag_e = False
    for i in range(evader_num - len(death), agent_num - len(death)):
        if tri[0] == i or tri[1] == i:
            flag_p = True
            break
        else:
            flag_p = False
    return flag_p and flag_e


def get_evader(tri):
    global death, evader_num
    for i in range(evader_num - len(death)):
        if tri[0] == i:
            return i
        elif tri[1] == i:
            return i


def get_pursuer(tri):
    global death, evader_num, agent_num
    for i in range(evader_num - len(death), agent_num - len(death)):
        if tri[0] == i:
            i = i - (evader_num - len(death))
            return i
        elif tri[1] == i:
            i = i - (evader_num - len(death))
            return i


def get_neighbor_nearest_evader(i, list_e, point):
    min_index = 0
    min_distance = 10000
    global death, evader_num
    for j in range(len(list_e)):
        distance = np.sqrt(np.sum(np.square(point[list_e[j]] - point[i + evader_num - len(death)])))
        if distance < min_distance:
            min_index = list_e[j]
            min_distance = distance
    return min_index


def get_theta(v):
    if v[0] > 0 and v[1] >= 0:
        return math.atan(v[1] / v[0])
    elif v[0] > 0 > v[1]:
        return math.atan(v[1] / v[0])
    elif v[0] == 0 and v[1] > 0:
        return np.pi / 2.0
    elif v[0] == 0 and v[1] < 0:
        return (-1) * np.pi / 2.0
    elif v[0] < 0 <= v[1]:
        return math.atan(v[1] / v[0]) + np.pi
    elif v[0] < 0 and v[1] < 0:
        return math.atan(v[1] / v[0]) - np.pi


def get_theta_diff(theta_s, theta_p):
    if theta_s >= 0 and theta_p >= 0:
        return theta_p - theta_s
    elif theta_s <= 0 and theta_p <= 0:
        return theta_p - theta_s
    elif (-1) * np.pi <= theta_s <= (-1. / 2.) * np.pi and np.pi >= theta_p >= 1. / 2. * np.pi:
        return theta_p + theta_s - 2 * np.pi
    elif (-1) * np.pi <= theta_p <= (-1. / 2.) * np.pi and np.pi >= theta_s >= 1. / 2. * np.pi:
        return 2 * np.pi - theta_p - theta_s
    elif (-1. / 2.) * np.pi <= theta_s <= 0 <= theta_p <= 1. / 2. * np.pi:
        return theta_p - theta_s
    elif (-1. / 2.) * np.pi <= theta_p <= 0 <= theta_s <= 1. / 2. * np.pi:
        return theta_p - theta_s


def get_unneighbor_nearest_evader(i, point):
    min_index = 0
    min_distance = 10000
    global death, evader_num
    for j in range(evader_num - len(death)):
        distance = np.sqrt(np.sum(np.square(point[j] - point[i + evader_num - len(death)])))
        if distance < min_distance:
            min_index = j
            min_distance = distance
    return min_index


def voronoi(points, bound):
    global agent_num, pursuer_num, evader_num
    circle = []
    tri_lines = []
    tri = Delaunay(points)
    for num in range(0, tri.simplices.shape[0]):
        plt.axis('equal')
        plt.axis('off')
        x, y, r = get_outer_circle(points[tri.simplices[num][0]], points[tri.simplices[num][1]],
                                   points[tri.simplices[num][2]])
        circle.append([x, y])
        tri.simplices[num].sort()
        tup = (tri.simplices[num][0], tri.simplices[num][1])
        tri_lines.append(tup)
        tup = (tri.simplices[num][0], tri.simplices[num][2])
        tri_lines.append(tup)
        tup = (tri.simplices[num][1], tri.simplices[num][2])
        tri_lines.append(tup)

    dic = dict()
    for i in range(len(tri_lines)):
        if tri_lines[i] in dic.keys():
            dic[tri_lines[i]].append(int(i) // int(3))
        else:
            dic[tri_lines[i]] = [int(i) // int(3)]

    voronoi_graph = dict()

    for key, value in dic.items():
        if len(value) == 2:
            x1, y1, x2, y2, flag = intersect(circle[value[0]], circle[value[1]], bound)
            voronoi_graph[key] = [[x1, y1], [x2, y2], flag]
        else:
            for i in range(0, 3):
                if tri.simplices[value[0]][i] != key[0] and tri.simplices[value[0]][i] != key[1]:
                    peak = [points[tri.simplices[value[0]][i]][0], points[tri.simplices[value[0]][i]][1]]
                    break
            if circle[value[0]][0] < bound[0] or circle[value[0]][0] > bound[1] or circle[value[0]][1] < \
                    bound[2] or circle[value[0]][1] > bound[3]:
                x1, y1 = circle[value[0]][0], circle[value[0]][1]
                x2, y2 = midline(points[key[0]], points[key[1]], peak, bound)
                flag = 0
            else:
                x1, y1, x2, y2, flag = intersect(circle[value[0]],
                                                 midline(points[key[0]], points[key[1]], peak, bound),
                                                 bound)
            voronoi_graph[key] = [[x1, y1], [x2, y2], flag]

    neighbor = []
    for i in range(agent_num - evader_num):
        neighbor.append([agent_num])
    unneighbor = []
    for tri_line in tri_lines:
        if is_tri(tri_line):
            if get_evader(tri_line) not in neighbor[get_pursuer(tri_line)]:
                if voronoi_graph[tri_line][2] != 0:
                    if voronoi_graph[tri_line][0][0] != voronoi_graph[tri_line][1][0] or voronoi_graph[tri_line][0][1] != voronoi_graph[tri_line][1][1]:
                        neighbor[get_pursuer(tri_line)].append(get_evader(tri_line))
    for i in range(agent_num - evader_num):
        if len(neighbor[i]) == 1:
            unneighbor.append(i)

    nearest_evader_index = []
    for i in range(agent_num - evader_num):
        nearest_evader_index.append(0)

    theta_v = []
    for i in range(agent_num - evader_num):
        if i in unneighbor:
            nearest_evader_index[i] = get_unneighbor_nearest_evader(i, points)
            direction = np.array([points[nearest_evader_index[i]][0] - points[i + evader_num - len(death)][0],
                                  points[nearest_evader_index[i]][1] - points[i + evader_num - len(death)][1]])
            direction = direction / np.sqrt(np.sum(np.square(direction)))
            theta = get_theta(direction)
            theta_v.append(theta)

            # points[i + evader_num - len(death)] = points[i + evader_num - len(death)] + direction
        else:
            nearest_evader_index[i] = get_neighbor_nearest_evader(i, neighbor[i][1:], points)
            mid = [(voronoi_graph[(nearest_evader_index[i], i + evader_num - len(death))][0][0] +
                    voronoi_graph[(nearest_evader_index[i], i + evader_num - len(death))][1][0]) / 2, (
                           voronoi_graph[(nearest_evader_index[i], i + evader_num - len(death))][0][1] +
                           voronoi_graph[(nearest_evader_index[i], i + evader_num - len(death))][1][1]) / 2]
            direction = np.array([mid[0] - points[i + evader_num - len(death)][0],
                                  mid[1] - points[i + evader_num - len(death)][1]])
            direction = direction / np.sqrt(np.sum(np.square(direction)))
            theta = get_theta(direction)
            theta_v.append(theta)
            # points[i + evader_num - len(death)] = points[i + evader_num - len(death)] + direction
    return theta_v


class Controller:
    __robot_name = ''
    __index = 0
    __capture_radius = 1

    def __init__(self, robot_name, index):
        print("class is setting up!")
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback, queue_size=10)
        self.__velpub = rospy.Publisher(robot_name + '/husky_velocity_controller/cmd_vel', queue_size=10)
        self.__setstate = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.__stamsg = SetModelStateRequest()
        self.__index = index
        self.__robot_name = robot_name

    def callback(self, msg):
        global pos_x, pos_y, linear_x, angular_z, roll, pitch, yaw, agent_num, Exit_Pos, death_num, death, flag_capture
        model_names = msg.name
        index, count, e_index, p_index = 0, 0, 0, 0

        for i in range(len(model_names)):
            if model_names[i] == self.__robot_name:
                index = i
                break

        roll[self.__index], pitch[self.__index], yaw[self.__index] = tf.transformations.euler_from_quaternion(
            [msg.pose[index].orientation.x, msg.pose[index].orientation.y,
             msg.pose[index].orientation.z, msg.pose[index].orientation.w])
        pos_x[self.__index] = msg.pose[index].position.x
        pos_y[self.__index] = msg.pose[index].position.y
        linear_x[self.__index] = msg.twist[index].linear.x
        angular_z[self.__index] = msg.twist[index].angular.z

        points = []

        for i in range(agent_num):
            if i not in death:
                points.append([pos_x[i], pos_y[i]])

        points = np.array(points)

        theta_v = voronoi(points, Exit_Pos)
        theta_r = yaw[self.__index]
        vel_msg = Twist()

        for i in range(agent_num - death_num):
            if np.sqrt(np.sum(np.square(points[0], points[i + 1]))) < 1:
                flag_capture = True
                break

        if flag_capture is not True:
            vel_msg.linear.x = 1.0
            vel_msg.linear.y = 0.0
            vel_msg.linear.z = 0.0
            vel_msg.angular.z = get_theta_diff(theta_r, theta_v[self.__index]) * 1
            vel_msg.angular.y = 0.0
            vel_msg.angular.x = 0.0
        else:
            vel_msg.linear.x = 0.0
            vel_msg.linear.y = 0.0
            vel_msg.linear.z = 0.0
            vel_msg.angular.z = 0.0
            vel_msg.angular.y = 0.0
            vel_msg.angular.x = 0.0
        self.__velpub.publish(vel_msg)


Controller("husky_alpha", 0), Controller("husky_gamma", 1), Controller("husky_delta", 2)

