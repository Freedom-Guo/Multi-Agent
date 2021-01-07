import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import random
import queue


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

    # 两条边的中垂线
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


##---------------------------------获取直线和边界的交点------------------------------------##
## a b c为直线解析式ax+by+c=0中的参数
## bound为边界的限制，例如矩阵：bound[0,1,2,3]分别表示x的最小最大值以及y的最小最大值
## (x1, y1)和(x2, y2)为交点的坐标
## flag表示寻找到交点的数量
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
        if (-c - a * bound[0]) / b <= bound[3] and (-c - a * bound[0]) / b >= bound[2]:
            # 线和x=bound[0]存在符合要求的交点
            if flag == 0:
                x1 = bound[0]
                y1 = (-c - a * bound[0]) / b
                flag = 1
            else:
                x2 = bound[0]
                y2 = (-c - a * bound[0]) / b
                flag = 2

        if (-c - a * bound[1]) / b <= bound[3] and (-c - a * bound[1]) / b >= bound[2]:
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

        if (-c - b * bound[2]) / a <= bound[1] and (-c - b * bound[2]) / a >= bound[0]:
            # 线和y=bound[2]存在符合要求的交点
            if flag == 0:
                y1 = bound[2]
                x1 = (-c - b * bound[2]) / a
                flag = 1
            else:
                y2 = bound[2]
                x2 = (-c - b * bound[2]) / a
                flag = 2

        if (-c - b * bound[3]) / a <= bound[1] and (-c - b * bound[3]) / a >= bound[0]:
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
    flag = 0
    C = [0, 0]
    if A[0] >= bound[0] and A[0] <= bound[1] and A[1] >= bound[2] and A[1] <= bound[3]:
        # A点在区域内部
        # print("A点在区域内部")
        if B[0] >= bound[0] and B[0] <= bound[1] and B[1] >= bound[2] and B[1] <= bound[3]:
            # B点在区域内
            flag = 1
            return A[0], A[1], B[0], B[1], flag
        else:
            # B点不在区域内
            flag = 1
            if (A[0] == B[0]):
                # AB的斜率不存在
                if (B[1] > bound[3]):
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
                if x1 >= min(A[0], B[0]) and x1 <= max(A[0], B[0]) and y1 >= min(A[1], B[1]) and y1 <= max(A[1], B[1]):
                    C[0] = x1
                    C[1] = y1
                else:
                    C[0] = x2
                    C[1] = y2
            return A[0], A[1], C[0], C[1], flag
    else:
        # A点不在区域内部
        if B[0] >= bound[0] and B[0] <= bound[1] and B[1] >= bound[2] and B[1] <= bound[3]:
            # B点在区域内
            flag = 1
            if (A[0] == B[0]):
                # AB的斜率不存在
                if (A[1] > bound[3]):
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
                if x1 >= min(A[0], B[0]) and x1 <= max(A[0], B[0]) and y1 >= min(A[1], B[1]) and y1 <= max(A[1], B[1]):
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


##-----------------------------------获取evader的策略-------------------------------------##
evader_strategy = queue.Queue()
death = []


def evader_controller(event):
    global evader_strategy
    if evader_strategy.qsize() < 10:
        # print("evader策略队列未满")
        evader_strategy.put([event.xdata, event.ydata])
    else:
        print("evader策略队列已满")
    return

##-----------------------------------其他功能函数------------------------------------------##

agent_num = 10
evader_num = 3

def is_evader(key_):
    global death, evader_num
    for i in range(evader_num - len(death)):
        if key_[0] == i or key_[1] == i:
            return True
    return False


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
    return (flag_p and flag_e)


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


def get_unneighbor_nearest_evader(i, point):
    min_index = 0
    min_distance = 10000
    global death, evader_num
    for j in range(evader_num - len(death)):
        distance = np.sqrt(np.sum(np.square(point[j] - point[i + evader_num - len(death)])))
        if (distance < min_distance):
            min_index = j
            min_distance = distance
    return min_index


## --------------------------------------Main Function----------------------------------##
# 绘制图的坐标X和Y
ax = []
ay = []
# 动态图
plt.ion()

# 更新步长
step = 2

# 设置区域边界
bounding_box = np.array([0., 100., 0., 100.])

# 绘制边界
plt.plot([bounding_box[0], bounding_box[0]], [bounding_box[2], bounding_box[3]], 'b-')
plt.plot([bounding_box[1], bounding_box[1]], [bounding_box[2], bounding_box[3]], 'b-')
plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[2], bounding_box[2]], 'b-')
plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[3], bounding_box[3]], 'b-')

points_pos = plt.ginput(agent_num)

# 输入起始坐标信息
points = []

for i in range(0, agent_num):
    ax.append(points_pos[i][0])
    ay.append(points_pos[i][1])
    points.append([ax[i], ay[i]])

points = np.array(points)

# 计算各个pursuer到evader的距离
# R[i][j]表示evader i到pursuer j的距离
Rc = np.zeros((evader_num, agent_num-evader_num))

for i in range(evader_num):
    for j in range(agent_num-evader_num):
        Rc[i][j] = np.sqrt(np.sum(np.square(points[i] - points[j + evader_num])))


def is_all_capture(distance):
    global death, evader_num, agent_num
    index = 0
    print(distance)
    for i in range(evader_num):
        if is_capture(distance[i][:]):
            if i not in death:
                death.append(i)
                for j in range(agent_num-evader_num):
                    distance[i][j] = 0.0
    for i in range(evader_num):
        if i in death:
            index += 1
    return (index != evader_num)


def is_capture(distance):
    global agent_num, evader_num
    for j in range(agent_num-evader_num):
        print("distance:")
        print(distance)
        if distance[j] <= 4:
            return True
    return False


# 获取鼠标键入
# d = [points[0][0], points[0][1]]
# cid = fig.canvas.mpl_connect('button_press_event', evader_controller)

while is_all_capture(Rc):

    # 生成Delaunay图，tri.simplices为三角形中顶点的索引，索引值为points中的索引值
    af_points = []
    for i in range(agent_num):
        if i not in death:
            af_points.append(points[i])

    af_points = np.array(af_points)

    tri = Delaunay(af_points)

    # 绘制pursuers和evader的点
    plt.clf()
    for i in range(evader_num):
        if i in death:
            plt.plot(ax[i], ay[i], 'kx', markersize=5)
        else:
            plt.plot(ax[i], ay[i], 'gp', markersize=5)
    plt.plot(ax[evader_num:], ay[evader_num:], 'ro', markersize=5)

    # 绘制边界
    plt.plot([bounding_box[0], bounding_box[0]], [bounding_box[2], bounding_box[3]], 'k-')
    plt.plot([bounding_box[1], bounding_box[1]], [bounding_box[2], bounding_box[3]], 'k-')
    plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[2], bounding_box[2]], 'k-')
    plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[3], bounding_box[3]], 'k-')

    # 声明外心和Delaunay三角网中三角形
    circle = []
    tri_lines = []

    # tri.simplices.shape[0]表示Delaunay三角网中三角形的数量
    # 获取三角形的外心和边的索引
    for num in range(0, tri.simplices.shape[0]):
        plt.axis('equal')
        plt.axis('off')
        x, y, r = get_outer_circle(af_points[tri.simplices[num][0]], af_points[tri.simplices[num][1]],
                                   af_points[tri.simplices[num][2]])
        circle.append([x, y])
        tri.simplices[num].sort()  # 对Delaunay三角形的顶点按照索引大小排序，方便对构造边的索引
        # 用边的顶点的索引构成边的元组，dic中key值不能为list，但可以是元组
        tup = (tri.simplices[num][0], tri.simplices[num][1])
        tri_lines.append(tup)
        tup = (tri.simplices[num][0], tri.simplices[num][2])
        tri_lines.append(tup)
        tup = (tri.simplices[num][1], tri.simplices[num][2])
        tri_lines.append(tup)

    # 构造边对应三角形索引的桶，遍历三角网中每个三角形的各边，获得边对应的三角形dic: (端点1，端点2):[三角形1，三角形2]
    # 三角形使用索引值为Delaunay()生成的三角网tri对每个三角形定义的索引值
    i = 0
    dic = dict()
    for tri_line in tri_lines:
        if tri_lines[i] in dic.keys():
            dic[tri_lines[i]].append(int(i) // int(3))
            i = i + 1
        else:
            dic[tri_lines[i]] = [int(i) // int(3)]
            i = i + 1

    # 构造Voronoi图，voronoi_graph对应关系为：具有相邻关系的两个智能体坐标索引-两智能体之间的Voronoi边
    voronoi_graph = dict()

    # 绘制Voronoi图
    # 遍历三角网中的每条边
    for key, value in dic.items():
        if len(value) == 2:
            # 该边是有公共三角形的边，则连接外心
            x1, y1, x2, y2, flag = intersect(circle[value[0]], circle[value[1]], bounding_box)
            # key值对应的两个智能体之间的Voronoi边
            voronoi_graph[key] = [[x1, y1], [x2, y2], flag]
            if flag:
                # flag为1 表示AB两点没有都在区域外面
                p1 = [x1, x2]
                p2 = [y1, y2]
                # 绘制Voronoi边
                if is_evader(key):
                    plt.plot(p1, p2, 'y-')
        else:
            # 没有公共边的三角形 连接外心和中垂线的交点
            # 获取三角形剩下的顶点
            for i in range(0, 3):
                if (tri.simplices[value[0]][i] != key[0] and tri.simplices[value[0]][i] != key[1]):
                    peak = [af_points[tri.simplices[value[0]][i]][0], af_points[tri.simplices[value[0]][i]][1]]
                    break
            # 获取Voronoi边的端点
            if circle[value[0]][0] < bounding_box[0] or circle[value[0]][0] > bounding_box[1] or circle[value[0]][1] < \
                    bounding_box[2] or circle[value[0]][1] > bounding_box[3]:
                x1, y1 = circle[value[0]][0], circle[value[0]][1]
                x2, y2 = midline(af_points[key[0]], af_points[key[1]], peak, bounding_box)
                flag = 0
            else:
                x1, y1, x2, y2, flag = intersect(circle[value[0]],
                                                 midline(af_points[key[0]], af_points[key[1]], peak, bounding_box),
                                                 bounding_box)
            # key值对应的两个智能体之间的Voronoi边
            voronoi_graph[key] = [[x1, y1], [x2, y2], flag]
            if flag:
                # flag为1 表示AB两点没有都在区域外面
                p1 = [x1, x2]
                p2 = [y1, y2]
                # 绘制Voronoi边
                if is_evader(key):
                    plt.plot(p1, p2, 'y-')

    print("voronoi_graph:")
    print(voronoi_graph)

    # 声明evader相邻和不相邻的pursuer的points索引
    neighbor = []
    for i in range(agent_num-evader_num):
        neighbor.append([agent_num])
    unneighbor = []

    # 获取pursuer相邻的evaders的points索引值并存入neighbor中
    for tri_line in tri_lines:
        if is_tri(tri_line):
            if get_evader(tri_line) not in neighbor[get_pursuer(tri_line)]:
                if voronoi_graph[tri_line][2] != 0:
                    if voronoi_graph[tri_line][0][0] != voronoi_graph[tri_line][1][0] or voronoi_graph[tri_line][0][
                        1] != voronoi_graph[tri_line][1][1]:
                        neighbor[get_pursuer(tri_line)].append(get_evader(tri_line))

    # 获取不存在evader相邻的pursuer的points索引值存入unneighbor中
    for i in range(agent_num-evader_num):
        if len(neighbor[i]) == 1:
            unneighbor.append(i)

    # 打印关键信息
    nearest_evader_index = []
    for i in range(agent_num-evader_num):
        nearest_evader_index.append(0)

    for i in range(agent_num-evader_num):
        if i in unneighbor:
            nearest_evader_index[i] = get_unneighbor_nearest_evader(i, af_points)
            direction = np.array([af_points[nearest_evader_index[i]][0] - af_points[i + evader_num - len(death)][0],
                                  af_points[nearest_evader_index[i]][1] - af_points[i + evader_num - len(death)][1]])
            direction = direction / np.sqrt(np.sum(np.square(direction)))
            af_points[i + evader_num - len(death)] = af_points[i + evader_num - len(death)] + direction
        else:
            nearest_evader_index[i] = get_neighbor_nearest_evader(i, neighbor[i][1:], af_points)
            mid = [(voronoi_graph[(nearest_evader_index[i], i + evader_num- len(death))][0][0] +
                    voronoi_graph[(nearest_evader_index[i], i + evader_num - len(death))][1][0]) / 2, (
                           voronoi_graph[(nearest_evader_index[i], i + evader_num - len(death))][0][1] +
                           voronoi_graph[(nearest_evader_index[i], i + evader_num - len(death))][1][1]) / 2]
            direction = np.array([mid[0] - af_points[i + evader_num - len(death)][0],
                                  mid[1] - af_points[i + evader_num - len(death)][1]])
            direction = direction / np.sqrt(np.sum(np.square(direction)))
            af_points[i + evader_num - len(death)] = af_points[i + evader_num - len(death)] + direction

    # evader运动策略 随机运动 最大速度为1

    # 键入evader的运动策略
    # if times%5 == 0:
    # evader_new = plt.ginput(1)
    # len_ee = np.sqrt(np.square(evader_new[0][0]-points[0][0])+np.square(evader_new[0][1]-points[0][1]))
    # Ve_x = (evader_new[0][0]-points[0][0]) / (10*len_ee)
    # Ve_y = (evader_new[0][1]-points[0][1]) / (10*len_ee)
    # times = times +1

    for i in range(evader_num - len(death)):
        Ve_x = random.uniform(-1, 1)
        Ve_y = np.sqrt(1 - Ve_x ** 2)
        D = [Ve_x, Ve_y]
        D = np.array(D)
        af_points[i] = af_points[i] + D

    af_index = 0
    be_points = []
    for i in range(agent_num):
        if i in death:
            be_points.append(points[i])
        else:
            be_points.append(af_points[af_index])
            af_index += 1

    points = np.array(be_points)

    for i in range(0, agent_num):
        if points[i][0] > bounding_box[1]:
            points[i][0] = bounding_box[1]

        if points[i][0] < bounding_box[0]:
            points[i][0] = bounding_box[0]

        if points[i][1] > bounding_box[3]:
            points[i][1] = bounding_box[3]

        if points[i][1] < bounding_box[2]:
            points[i][1] = bounding_box[2]

        ax[i] = points[i][0]
        ay[i] = points[i][1]


    for i in range(evader_num):
        for j in range(agent_num-evader_num):
            Rc[i][j] = np.sqrt(np.sum(np.square(points[i] - points[j + evader_num])))

    plt.pause(0.01)
    plt.ioff

plt.clf()
plt.axis('equal')
plt.axis('off')
plt.plot(ax[:evader_num], ay[:evader_num], 'kx', markersize=5)
plt.plot(ax[evader_num:], ay[evader_num:], 'ro', markersize=5)
plt.plot([bounding_box[0], bounding_box[0]], [bounding_box[2], bounding_box[3]], 'k-')
plt.plot([bounding_box[1], bounding_box[1]], [bounding_box[2], bounding_box[3]], 'k-')
plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[2], bounding_box[2]], 'k-')
plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[3], bounding_box[3]], 'k-')
plt.pause(3)
plt.ioff
print("Capture Successfully!")



