import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import operator
import random

agent_num = 10
evader_num = 3
capture_radius = 5


def distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt(np.sum(np.square(p1 - p2)))


def init_judge(points, E, rc):
    global agent_num, evader_num
    defender = []
    for i in range(evader_num):
        for j in range(evader_num, agent_num):
            # judge the distance of every pursuer and evader is over capture radius
            # if less, break the loop of pursuers node
            if distance(points[i], points[j]) > rc:
                # judge the pursuer is closer to E than the evader, means the distance between E and p is closer
                if distance_diff(points[j], points[i], E[0]) > 0:
                    if distance_diff(points[j], points[i], E[1]) > 0:
                        # judge the pursuer is in defender list
                        if j not in defender:
                            defender.append(j)
                            break
            else:
                break
    if len(defender) == 3:
        return True
    else:
        return False


def distance_diff(p, e, E):
    return distance(e, E) - distance(p, E)


def distance_pek(points, E):
    global agent_num, evader_num, death
    pursuer_num = agent_num - evader_num
    distance_difference = np.zeros((pursuer_num, evader_num-len(death), len(E)))
    for i in range(pursuer_num):
        for j in range(evader_num-len(death)):
            for k in range(len(E)):
                distance_difference[i][j][k] = distance_diff(points[evader_num + i], points[j], E[k])
    return np.array(distance_difference)


def is_all_capture(dis):
    global death, evader_num, agent_num
    pursuers_num = agent_num - evader_num
    index = 0
    for i in range(evader_num):
        if is_capture(dis[i][:]):
            if i not in death:
                death.append(i)
                for j in range(pursuers_num):
                    dis[i][j] = 0.0
    for i in range(evader_num):
        if i in death:
            index += 1
    return index != evader_num


def is_capture(dis):
    global agent_num, evader_num, capture_radius
    pursuers_num = agent_num-evader_num
    for k in range(pursuers_num):
        if dis[k] <= capture_radius:
            return True
    return False


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


def get_intersect_point(a, b, c, bound):
    # 初始化
    flag = 0
    x1 = y1 = x2 = y2 = 0

    if b == 0:
        # 斜率不存在
        x1 = x2 = -c / a
        y1 = bound[2]
        y2 = bound[3]
    else:
        # 斜率存在
        if ((-c - a * bound[0]) / b <= bound[3]) and ((-c - a * bound[0]) / b >= bound[2]):
            # 线和x=bound[0]存在符合要求的交点
            if flag == 0:
                x1 = bound[0]
                y1 = (-c - a * bound[0]) / b
                flag = 1
            else:
                x2 = bound[0]
                y2 = (-c - a * bound[0]) / b
                flag = 2

        if ((-c - a * bound[1]) / b <= bound[3]) and ((-c - a * bound[1]) / b >= bound[2]):
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

        if (-c - b * bound[2]) / a <= bound[1] and ((-c - b * bound[2]) / a >= bound[0]):
            # 线和y=bound[2]存在符合要求的交点
            if flag == 0:
                y1 = bound[2]
                x1 = (-c - b * bound[2]) / a
                flag = 1
            else:
                y2 = bound[2]
                x2 = (-c - b * bound[2]) / a
                flag = 2

        if ((-c - b * bound[3]) / a <= bound[1]) and ((-c - b * bound[3]) / a >= bound[0]):
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

    return x1, y1, x2, y2


def intersect(A, B, bound):
    flag = 0
    C = [0, 0]
    if (A[0] >= bound[0]) and (A[0] <= bound[1]) and (A[1] >= bound[2]) and (A[1] <= bound[3]):
        # A点在区域内部
        if (B[0] >= bound[0]) and (B[0] <= bound[1]) and (B[1] >= bound[2]) and (B[1] <= bound[3]):
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
                x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
                if (x1 >= min(A[0], B[0])) and (x1 <= max(A[0], B[0])) and (y1 >= min(A[1], B[1])) and (y1 <= max(A[1], B[1])):
                    C[0] = x1
                    C[1] = y1
                else:
                    C[0] = x2
                    C[1] = y2
            return A[0], A[1], C[0], C[1], flag
    else:
        # A点不在区域内部
        if (B[0] >= bound[0]) and (B[0] <= bound[1]) and (B[1] >= bound[2]) and (B[1] <= bound[3]):
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
                x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
                if (x1 >= min(A[0], B[0])) and (x1 <= max(A[0], B[0])) and (y1 >= min(A[1], B[1])) and (y1 <= max(A[1], B[1])):
                    C[0] = x1
                    C[1] = y1
                else:
                    C[0] = x2
                    C[1] = y2
            return B[0], B[1], C[0], C[1], flag
        else:
            flag = 1
            if A[0] == B[0]:
                flag = 0
                return A[0], A[1], B[0], B[1], flag
            else:
                a = A[1] - B[1]
                b = B[0] - A[0]
                c = B[1] * A[0] - A[1] * B[0]
                x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
                return x1, y1, x2, y2, flag


def IsIntersec(p1, p2, p3, p4):
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = p1[0] * p2[1] - p2[0] * p1[1]
    if (a * p3[0] + b * p3[1] + c) * (a * p4[0] + b * p4[1] + c) <= 0:
        return 1
    else:
        return 0


def midline(A, B, C, bound):
    a = 2 * (B[0] - A[0])
    b = 2 * (B[1] - A[1])
    c = A[0] ** 2 - B[0] ** 2 + A[1] ** 2 - B[1] ** 2
    x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
    D = [x1, y1]
    if IsIntersec(A, B, C, D):
        D = [x1, y1]
    else:
        D = [x2, y2]
    return D


ax = []
ay = []

plt.ion()
fig = plt.figure()

bounding_box = np.array([0., 100., 0., 100.])
E_ = [[0., 45.], [0., 55.]]
E_ = np.array(E_)

plt.plot([bounding_box[0], E_[0][0]], [bounding_box[2], E_[0][1]], 'k-')
plt.plot([E_[1][0], bounding_box[0]], [E_[1][1], bounding_box[3]], 'k-')
plt.plot([bounding_box[1], bounding_box[1]], [bounding_box[2], bounding_box[3]], 'k-')
plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[2], bounding_box[2]], 'k-')
plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[3], bounding_box[3]], 'k-')

points = []

while True:
    points_pos = plt.ginput(agent_num)
    if init_judge(points_pos, E_, capture_radius):
        break

points = np.array(points_pos)
pursuer_num = agent_num - evader_num

evader_to_pursuer = np.zeros((evader_num, pursuer_num))
for i in range(evader_num):
    for j in range(pursuer_num):
        evader_to_pursuer[i][j] = distance(points[i], points[evader_num+j])

death = []
defender_1 = []
defender_2 = []

min = 0.5

while is_all_capture(evader_to_pursuer):
    af_points = []
    for i in range(agent_num):
        if i not in death:
            af_points.append(points[i])

    af_points = np.array(af_points)

    tri = Delaunay(af_points)

    pursuer_evader_E_diff = distance_pek(af_points, E_)

    dic_1 = dict()
    dic_2 = dict()

    for j in range(evader_num-len(death)):
        for i in range(pursuer_num):
            if pursuer_evader_E_diff[i][j][0] <= min:
                dic_1[i] = pursuer_evader_E_diff[i][j][0]
            if pursuer_evader_E_diff[i][j][1] <= min:
                dic_2[i] = pursuer_evader_E_diff[i][j][1]
        dic_1_temp = sorted(dic_1.items(), key=operator.itemgetter(1))
        dic_2_temp = sorted(dic_2.items(), key=operator.itemgetter(1))
        temp1 = dic_1_temp[len(dic_1_temp)-1][1]
        temp2 = dic_2_temp[len(dic_2_temp)-1][1]
        if temp1 not in defender_1:
            defender_1.append(temp1)
