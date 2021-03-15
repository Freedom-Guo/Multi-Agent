import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


## 函数传入参数：
## 机器人的数量: agent_num （int整型）
## 机器人的坐标: agent_pos = [[e_x, e_y], [p1_x, p2_y], ...] 索引值0对应被抓捕者坐标 （np数组形式）
## 边界坐标: bound = [[left_down_x, left_down_y], [right_up_x, right_up_y]] (np数组形式)
## 逐步着最大速度值: max_v
## 决策频率: frequency

## 函数输出:
## 抓捕者的速度: pursuer_v = [[p1_v_x, p1_v_y], ...] 索引值n对应n-1号抓捕者

def pursuer_decision(agent_num, agent_pos, bound, max_v, frequency):
    tri = Delaunay(agent_pos)
    circle = []
    tri_lines = []
    bounding_box = [bound[0][0], bound[1][0], bound[0][1], bound[1][1]]

    def get_outer_circle(A, B, C):
        xa, ya = A[0], A[1]
        xb, yb = B[0], B[1]
        xc, yc = C[0], C[1]

        xab, yab = (xa + xb) / 2.0, (ya + yb) / 2.0
        xbc, ybc = (xb + xc) / 2.0, (yb + yc) / 2.0

        if (xb != xa):
            kab = (yb - ya) / (xb - xa)
        else:
            kab = None

        if (xc != xb):
            kbc = (yc - yb) / (xc - xb)
        else:
            kbc = None

        if (kab != None):
            ab = np.arctan(kab)
        else:
            ab = np.pi / 2

        if (kbc != None):
            bc = np.arctan(kbc)
        else:
            bc = np.pi / 2

        if (ab == 0):
            kabm = None
            b1 = 0
            x = xab
        else:
            kabm = np.tan(ab + np.pi / 2)
            b1 = yab * 1.0 - xab * kabm * 1.0

        if (bc == 0):
            kbcm = None
            b2 = 0
            x = xbc
        else:
            kbcm = np.tan(bc + np.pi / 2)
            b2 = ybc * 1.0 - xbc * kbcm * 1.0

        if (kabm != None and kbcm != None):
            x = (b2 - b1) * 1.0 / (kabm - kbcm)

        if (kabm != None):
            y = kabm * x * 1.0 + b1 * 1.0
        else:
            y = kbcm * x * 1.0 + b2 * 1.0

        r = np.sqrt((x - xa) ** 2 + (y - ya) ** 2)
        return (x, y, r)

    for num in range(0, tri.simplices.shape[0]):
        plt.axis('equal')
        plt.axis('off')
        x, y, r = get_outer_circle(agent_pos[tri.simplices[num][0]], agent_pos[tri.simplices[num][1]],
                                   agent_pos[tri.simplices[num][2]])
        circle.append([x, y])
        tri.simplices[num].sort()
        tup = (tri.simplices[num][0], tri.simplices[num][1])
        tri_lines.append(tup)
        tup = (tri.simplices[num][0], tri.simplices[num][2])
        tri_lines.append(tup)
        tup = (tri.simplices[num][1], tri.simplices[num][2])
        tri_lines.append(tup)

    i = 0
    dic = dict()
    for tri_line in tri_lines:
        if tri_line in dic.keys():
            dic[tri_lines[i]].append(int(i) // int(3))
            i = i + 1
        else:
            dic[tri_lines[i]] = [int(i) // int(3)]
            i = i + 1

    voronoi_graph = dict()

    def get_intersect_point(a, b, c, bound):
        flag = 0
        x1 = y1 = x2 = y2 = 0

        if b == 0:
            x1 = x2 = -c / a
            y1 = bound[2]
            y2 = bound[3]
        else:
            # 斜率存在
            if (-c - a * bound[0]) / b <= bound[3] and (-c - a * bound[0]) / b >= bound[2]:
                # print("线和x=bound[0]存在符合要求的交点")
                if flag == 0:
                    x1 = bound[0]
                    y1 = (-c - a * bound[0]) / b
                    flag = 1
                else:
                    x2 = bound[0]
                    y2 = (-c - a * bound[0]) / b
                    flag = 2

            if (-c - a * bound[1]) / b <= bound[3] and (-c - a * bound[1]) / b >= bound[2]:
                # print("线和x=bound[1]存在符合要求的交点")
                if flag == 0:
                    x1 = bound[1]
                    y1 = (-c - a * bound[1]) / b
                    flag = 1
                else:
                    # 找到过符合要求的交点
                    x2 = bound[1]
                    y2 = (-c - a * bound[1]) / b
                    flag = 2

            if (-c - b * bound[2]) / a <= bound[1] and (-c - b * bound[2]) / a >= bound[0]:
                # print("线和y=bound[2]存在符合要求的交点")
                if flag == 0:
                    y1 = bound[2]
                    x1 = (-c - b * bound[2]) / a
                    flag = 1
                else:
                    y2 = bound[2]
                    x2 = (-c - b * bound[2]) / a
                    flag = 2

            if (-c - b * bound[3]) / a <= bound[1] and (-c - b * bound[3]) / a >= bound[0]:
                # print("线和y=bound[3]存在符合要求的交点")
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

        return flag, x1, y1, x2, y2

    def intersect(A, B, bound):
        C = [0, 0]
        if A[0] >= bound[0] and A[0] <= bound[1] and A[1] >= bound[2] and A[1] <= bound[3]:
            if B[0] >= bound[0] and B[0] <= bound[1] and B[1] >= bound[2] and B[1] <= bound[3]:
                flag = 1;
                return A[0], A[1], B[0], B[1], flag
            else:
                flag = 1
                if (A[0] == B[0]):
                    if (B[1] > bound[3]):
                        x = A[0]
                        y = bound[3]
                    else:
                        x = A[0]
                        y = bound[2]
                    C[0] = x
                    C[1] = y
                else:
                    a = A[1] - B[1]
                    b = B[0] - A[0]
                    c = B[1] * A[0] - A[1] * B[0]
                    num, x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
                    if x1 >= min(A[0], B[0]) and x1 <= max(A[0], B[0]) and y1 >= min(A[1], B[1]) and y1 <= max(A[1],
                                                                                                               B[1]):
                        C[0] = x1
                        C[1] = y1
                    else:
                        C[0] = x2
                        C[1] = y2
                return A[0], A[1], C[0], C[1], flag
        else:
            if B[0] >= bound[0] and B[0] <= bound[1] and B[1] >= bound[2] and B[1] <= bound[3]:
                flag = 1
                if (A[0] == B[0]):
                    if (A[1] > bound[3]):
                        x = B[0]
                        y = bound[3]
                    else:
                        x = B[0]
                        y = bound[2]
                    C = [x, y]
                else:
                    a = A[1] - B[1]
                    b = B[0] - A[0]
                    c = B[1] * A[0] - A[1] * B[0]
                    num, x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
                    if x1 >= min(A[0], B[0]) and x1 <= max(A[0], B[0]) and y1 >= min(A[1], B[1]) and y1 <= max(A[1],
                                                                                                               B[1]):
                        C[0] = x1
                        C[1] = y1
                    else:
                        C[0] = x2
                        C[1] = y2
                return B[0], B[1], C[0], C[1], flag
            else:
                flag = 0
                if (A[0] == B[0]):
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

    def IsIntersec(p1, p2, p3, p4):
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        c = p2[0] * p1[1] - p1[0] * p2[1]
        # print(a, b, c)
        if (a * p3[0] + b * p3[1] + c) * (a * p4[0] + b * p4[1] + c) <= 0:
            return 1
        else:
            return 0

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

    for key, value in dic.items():
        if len(value) == 2:
            x1, y1, x2, y2, flag = intersect(circle[value[0]], circle[value[1]], bounding_box)
            voronoi_graph[key] = [[x1, y1], [x2, y2], flag]
        else:
            for i in range(0, 3):
                if (tri.simplices[value[0]][i] != key[0] and tri.simplices[value[0]][i] != key[1]):
                    peak = [agent_pos[tri.simplices[value[0]][i]][0], agent_pos[tri.simplices[value[0]][i]][1]]
                    break
            if circle[value[0]][0] < bounding_box[0] or circle[value[0]][0] > bounding_box[1] or circle[value[0]][
                1] < \
                    bounding_box[2] or circle[value[0]][1] > bounding_box[3]:
                x1, y1, x2, y2, flag = intersect(circle[value[0]],
                                                 midline(agent_pos[key[0]], agent_pos[key[1]], peak, bounding_box),
                                                 bounding_box)
            else:
                x1, y1 = circle[value[0]][0], circle[value[0]][1]
                x2, y2 = midline(agent_pos[key[0]], agent_pos[key[1]], peak, bounding_box)
                flag = 1
            voronoi_graph[key] = [[x1, y1], [x2, y2], flag]

    neighbor = []
    unneighbor = []

    for tri_line in tri_lines:
        if (tri_line[0] == 0 or tri_line[1] == 0):
            if tri_line[1] + tri_line[0] not in neighbor:
                if voronoi_graph[tri_line][2] != 0:
                    if voronoi_graph[tri_line][0][0] != voronoi_graph[tri_line][1][0] or voronoi_graph[tri_line][0][
                        1] != voronoi_graph[tri_line][1][1]:
                        neighbor.append(tri_line[1] + tri_line[0])

    for i in range(1, agent_num):
        if i not in neighbor:
            unneighbor.append(i)

    vp = []
    for i in range(1, agent_num):
        if i in neighbor:
            mid = np.array([(voronoi_graph[(0, i)][0][0] + voronoi_graph[(0, i)][1][0]) / 2,
                            (voronoi_graph[(0, i)][0][1] + voronoi_graph[(0, i)][1][1]) / 2])
            vp.append((mid - agent_pos[i]) * max_v / (np.sqrt(np.sum(np.square(mid - agent_pos[i]))) * frequency) )
        else:
            vp.append((agent_pos[0] - agent_pos[i]) * max_v / (np.sqrt(np.sum(np.square(agent_pos[0] - agent_pos[i]))) * frequency) )

    pursuer_v = np.array(vp)

    return pursuer_v
