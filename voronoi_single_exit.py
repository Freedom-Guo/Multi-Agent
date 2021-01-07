import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import random
import queue


##----------------------------判定键入evader和pursuers的坐标是否满足要求---------------------##
## 判断准则：存在一个Pi到E上的任意点的距离小于e到E的距离，且任意Pi到e的距离都大于Rc
## evader默认为键入点中索引值为0的点
## param:
## points 键入的点的坐标数组; E_出口E的两个端点
## return:
## result 键入的点的坐标是否满足要求 0-不满足 1-满足;
def initial_judge(points, E_, Rc):
    print("对初始坐标验证中......")
    result = 0
    for i in range(1, 3):
        if np.square(points[i][0] - points[0][0]) + np.square(points[i][1] - points[0][1]) > Rc ** 2:
            if 2 * (points[i][1] - points[0][1]) * E_[0][1] + np.square(points[0][1]) - np.square(
                    points[i][1]) + np.square(points[0][0]) - np.square(points[i][0]) > 0:
                if 2 * (points[i][1] - points[0][1]) * E_[1][1] + np.square(points[0][1]) - np.square(
                        points[i][1]) + np.square(points[0][0]) - np.square(points[i][0]) > 0:
                    # 存在一个Pi到E上的任意点的距离小于e到E的距离，置1，继续循环
                    result = 1
        else:
            # 存在Pi到e的距离小于Rc，不满足条件，置0，并结束循环
            result = 0
            break
    if result:
        print("初始坐标满足要求")
    else:
        print("初始坐标不满足要求，请重新键入")
    return result


##------------------------------------分配defender角色------------------------------------##
## 获取defender的坐标，默认为距离E最近的点
## param:
## points 键入的点的坐标数组; E_出口E的两个端点
## return:
## defender_index defender在键入的点集中的索引值
def get_defender(points, E_):
    min_distance = 100
    min_i = 0
    for i in range(1, 3):
        if points[i][1] <= E_[1][1] and points[i][1] >= E_[0][1]:
            distance = points[i][0] ** 2
            if distance <= min_distance:
                min_distance = distance
                min_i = i
        else:
            distance = min((np.square(points[i][0]) + np.square(points[i][1] - E_[0][1])),
                           (np.square(points[i][0]) + np.square(points[i][1] - E_[1][1])))
            if distance <= min_distance:
                min_distance = distance
                min_i = i
    return min_i


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

    return x1, y1, x2, y2


##-----------------------------获取Voronoit图中需要连接的两点-------------------------------##
## A B为需要连接的两点，因为需要考虑其中点在边界以外，这个需要截取线段在边界内部的部分，如果都在外面则舍弃
## flag 1表示A、B两点不是都在边界以外 0表示A、B两点都在边界以外
## C表示截取线段中位于边界上的端点
def intersect(A, B, bound):
    flag = 0
    C = [0, 0]
    if A[0] >= bound[0] and A[0] <= bound[1] and A[1] >= bound[2] and A[1] <= bound[3]:
        # A点在区域内部
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
                x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
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
                x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
                if x1 >= min(A[0], B[0]) and x1 <= max(A[0], B[0]) and y1 >= min(A[1], B[1]) and y1 <= max(A[1], B[1]):
                    C[0] = x1
                    C[1] = y1
                else:
                    C[0] = x2
                    C[1] = y2
            return B[0], B[1], C[0], C[1], flag
        else:
            flag = 1
            if (A[0] == B[0]):
                flag = 0
                return A[0], A[1], B[0], B[1], flag
            else:
                a = A[1] - B[1]
                b = B[0] - A[0]
                c = B[1] * A[0] - A[1] * B[0]
                x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
                return x1, y1, x2, y2, flag


##----------------------------------判断两点是否位于直线异侧--------------------------------##
## p1和p2为直线上的两点，p3和p4是需要进行判断的两点
## a b c为直线p1p2解析式ax+by+c=0中的参数
def IsIntersec(p1, p2, p3, p4):
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = p1[0] * p2[1] - p2[0] * p1[1]
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
    x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
    D = [x1, y1]
    if IsIntersec(A, B, C, D):
        D = [x1, y1]
    else:
        D = [x2, y2]
    return D


##-----------------------------------获取li的两个端点-------------------------------------##
def get_l_point(A, B, C, D):
    # A-交点1 B-交点2 C-evader D-pursuer
    a1 = A[1] - B[1]
    b1 = B[0] - A[0]
    c1 = B[1] * A[0] - B[0] * A[1]
    a2 = C[1] - D[1]
    b2 = D[0] - C[0]
    c2 = D[1] * C[0] - D[0] * C[1]
    d = a1 * b2 - a2 * b1
    x1 = (b1 * c2 - b2 * c1) / d
    y1 = (a2 * c1 - a1 * c2) / d
    v1 = [D[0] - A[0], D[1] - A[1]]
    v2 = [D[0] - C[0], D[1] - C[1]]
    if np.cross(v1, v2) > 0:
        x2 = A[0]
        y2 = A[1]
    else:
        x2 = B[0]
        y2 = B[1]
    return x1, y1, x2, y2


##-----------------------------------获取evader的策略-------------------------------------##
evader_strategy = queue.Queue()


def evader_controller(event):
    global evader_strategy
    if evader_strategy.qsize() < 10:
        print("evader策略队列未满")
        evader_strategy.put([event.xdata, event.ydata])
        print(event.xdata, event.ydata)
    else:
        print("evader策略队列已满")
    return


## ------------------------------------Main Function------------------------------------##
# 绘制图的坐标X和Y
ax = []
ay = []

# 动态图
plt.ion()
fig = plt.figure()

# 更新步长
step = 5

# 设置区域边界
bounding_box = np.array([0., 10., 0., 10.])
E_ = [[0., 4.], [0., 6.]]
E_ = np.array(E_)

# 绘制边界
plt.plot([bounding_box[0], E_[0][0]], [bounding_box[2], E_[0][1]], 'k-')
plt.plot([E_[1][0], bounding_box[0]], [E_[1][1], bounding_box[3]], 'k-')
plt.plot([bounding_box[1], bounding_box[1]], [bounding_box[2], bounding_box[3]], 'k-')
plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[2], bounding_box[2]], 'k-')
plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[3], bounding_box[3]], 'k-')

points = []

Rc_limt = 0.5

while True:
    points_pos = plt.ginput(3)
    print("初始坐标：")
    print(points_pos)
    if initial_judge(points_pos, E_, Rc_limt):
        j = 0
        # evader i=0
        ax.append(points_pos[0][0])
        ay.append(points_pos[0][1])
        points.append([ax[j], ay[j]])
        j = j + 1
        # defender i=1
        defender_i = get_defender(points_pos, E_)
        ax.append(points_pos[defender_i][0])
        ay.append(points_pos[defender_i][1])
        points.append([ax[j], ay[j]])
        j = j + 1
        # pursuers i=2.3
        for i in range(1, 3):
            if i != defender_i:
                ax.append(points_pos[i][0])
                ay.append(points_pos[i][1])
                points.append([ax[j], ay[j]])
                j = j + 1
        break

points = np.array(points)

# 计算各个pursuer到evader的距离
Rc = []

for i in range(1, 3):
    Rc.append(np.sqrt(np.sum(np.square(points[0] - points[i]))))

Rc = np.array(Rc)

D = [0, 0]

d = [points[0][0], points[0][1]]

# 获取鼠标键入
cid = fig.canvas.mpl_connect('button_press_event', evader_controller)

while Rc[0] > 0.5 and Rc[1] > 0.5 and points[0][0] - D[0] >= 0:
    print("-------------------------------test----------------------------")
    # print(Rc[0], Rc[1])
    print("点的坐标：")
    print(points)
    # 计算Rc
    for i in range(0, 2):
        Rc[i] = (np.sqrt(np.sum(np.square(points[0] - points[i + 1]))))
        print(Rc[i])

    # 默认0号point就是evader，1号point是defender
    print("evader:")
    print(points[0])

    print("defender:")
    print(points[1])

    # 绘制pursuers和evader的点
    plt.clf()
    plt.plot(ax[0], ay[0], 'gp')
    plt.plot(ax[1], ay[1], 'b*')
    plt.plot(ax[2:], ay[2:], 'ro')

    # 绘制边界
    plt.plot([bounding_box[0], E_[0][0]], [bounding_box[2], E_[0][1]], 'k-')
    plt.plot([E_[1][0], bounding_box[0]], [E_[1][1], bounding_box[3]], 'k-')
    plt.plot([bounding_box[1], bounding_box[1]], [bounding_box[2], bounding_box[3]], 'k-')
    plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[2], bounding_box[2]], 'k-')
    plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[3], bounding_box[3]], 'k-')

    s = []

    # 计算s1和s2
    for i in range(0, 2):
        s.append(np.sqrt(np.sum(np.square(E_[i] - points[1]))) - np.sqrt(np.sum(np.square(E_[i] - points[0]))))

    # Sj+的宽度定义
    wid = 0.5

    defender_flag = -1

    # 制定defender的策略
    if s[0] < 0 and s[1] < 0:
        # 位于Dp内部
        defender_flag = 1
    elif s[0] < wid and s[0] >= 0:
        # 位于S1和S1+上
        defender_flag = 0
        uj_ = (E_[0] - points[1]) / (step * np.sqrt(np.sum(np.square(E_[0] - points[1]))))
    elif s[1] < wid and s[1] >= 0:
        # 位于S2和S2+上
        defender_flag = 0
        uj_ = (E_[1] - points[1]) / (step * np.sqrt(np.sum(np.square(E_[1] - points[1]))))


    Ve_ = []
    # 三点共线的时候
    if points[0][0] * points[1][1] - points[1][0] * points[0][1] + points[1][0] * points[2][1] - points[2][0] * \
            points[1][1] + points[2][0] * points[0][1] - points[0][0] * points[2][1] == 0:
        print("此时三点共线")
        for i in range(0, 2):
            Ve_.append((points[0] - points[i + 1]) / (step * Rc[i]))
        if defender_flag == 0:
            print("defender在Sj U Sj+内，执行防御策略")
            points[1] = points[1] + uj_
            points[2] = points[2] + Ve_[1]
        else:
            print("defender在Dp内部，执行追捕策略")
            points[i + 1] = points[i + 1] + Ve_[i]
    else:
        # 生成Delaunay图，tri.simplices为三角形中顶点的索引，索引值为points中的索引值
        tri = Delaunay(points)

        # 打印三角形的顶点索引
        # print("Delaunay三角形的顶点索引")
        # print(tri.simplices)

        # 声明外心和Delaunay三角网中三角形
        circle = []
        tri_lines = []

        # 获取三角形的外心和边的索引
        for num in range(0, tri.simplices.shape[0]):
            print(num)
            plt.axis('equal')
            plt.axis('off')
            x, y, r = get_outer_circle(points[tri.simplices[num][0]], points[tri.simplices[num][1]],
                                       points[tri.simplices[num][2]])
            circle.append([x, y])
            tri.simplices[num].sort()  # 对Delaunay三角形的顶点按照索引大小排序，方便对构造边的索引
            # print("Delaunay三角形的顶点坐标：")
            # print(points[tri.simplices[num][0]], points[tri.simplices[num][1]], points[tri.simplices[num][2]])
            # 用边的顶点的索引构成边的元组，dic中key值不能为list，但可以是元组
            tup = (tri.simplices[num][0], tri.simplices[num][1])
            tri_lines.append(tup)
            tup = (tri.simplices[num][0], tri.simplices[num][2])
            tri_lines.append(tup)
            tup = (tri.simplices[num][1], tri.simplices[num][2])
            tri_lines.append(tup)

        # print("Delaunay所有三角形的边的索引对：")
        # print(tri_lines)

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

        print("边-三角形对应情况：")
        print(dic)

        # 构造Voronoi图，voronoi_graph对应关系为：具有相邻关系的两个智能体坐标索引-两智能体之间的Voronoi边
        voronoi_graph = dict()

        # 绘制Voronoi图
        # 遍历三角网中的每条边
        for key, value in dic.items():
            print(key)
            print("边对应的三角形：")
            print(value)
            if len(value) == 2:
                # 该边是有公共三角形的边，则连接外心
                # print(circle[value[0]], circle[value[1]])
                x1, y1, x2, y2, flag = intersect(circle[value[0]], circle[value[1]], bounding_box)
                # key值对应的两个智能体之间的Voronoi边
                voronoi_graph[key] = [[x1, y1], [x2, y2]]
                if flag:
                    # flag为1 表示AB两点没有都在区域外面
                    p1 = [x1, x2]
                    p2 = [y1, y2]
                    # print("Voronoi边的端点：")
                    # print(p1, p2)
                # 绘制Voronoi边
                plt.plot(p1, p2, 'y-')
            else:
                # 没有公共边的三角形 连接外心和中垂线的交点
                print("没有公共边三角形")
                # 获取三角形剩下的顶点
                for i in range(0, 3):
                    if (tri.simplices[value[0]][i] != key[0] and tri.simplices[value[0]][i] != key[1]):
                        peak = [points[tri.simplices[value[0]][i]][0], points[tri.simplices[value[0]][i]][1]]
                        break
                # 获取Voronoi边的端点
                if circle[value[0]][0] < bounding_box[0] or circle[value[0]][0] > bounding_box[1] or circle[value[0]][
                    1] < bounding_box[2] or circle[value[0]][1] > bounding_box[3]:
                    x1, y1 = circle[value[0]][0], circle[value[0]][1]
                    x2, y2 = midline(points[key[0]], points[key[1]], peak, bounding_box)
                    flag = 0
                else:
                    x1, y1, x2, y2, flag = intersect(circle[value[0]],
                                                     midline(points[key[0]], points[key[1]], peak, bounding_box),
                                                     bounding_box)
                # key值对应的两个智能体之间的Voronoi边
                voronoi_graph[key] = [[x1, y1], [x2, y2]]
                if flag:
                    # flag为1 表示AB两点没有都在区域外面
                    p1 = [x1, x2]
                    p2 = [y1, y2]
                    # print("Voronoi边的端点：")
                    # print(p1, p2)
                # 绘制Voronoi边
                plt.plot(p1, p2, 'y-')

        # 声明evader相邻和不相邻的pursuer的points索引
        neighbor = []
        unneighbor = []

        # 获取evader相邻的pursuers的points索引值并存入neighbor中
        for tri_line in tri_lines:
            if (tri_line[0] == 0 or tri_line[1] == 0):
                if tri_line[1] + tri_line[0] not in neighbor:
                    if voronoi_graph[tri_line][0][0] != voronoi_graph[tri_line][1][0] or voronoi_graph[tri_line][0][
                        1] != voronoi_graph[tri_line][1][1]:
                        neighbor.append(tri_line[1] + tri_line[0])

        # 获取evader不相邻的pursuers的points索引值并存入unneighbor中
        for i in range(1, 3):
            if i not in neighbor:
                unneighbor.append(i)

        vp = []

        for i in range(1, 3):
            if i in neighbor:
                if i == 1:
                    if defender_flag:
                        mid = np.array([(voronoi_graph[(0, i)][0][0] + voronoi_graph[(0, i)][1][0]) / 2,
                                    (voronoi_graph[(0, i)][0][1] + voronoi_graph[(0, i)][1][1]) / 2])
                        vp.append((mid - points[i]) / (step * np.sqrt(np.sum(np.square(mid - points[i])))))
                    else:
                        vp.append(uj_)
                else:
                    mid = np.array([(voronoi_graph[(0, i)][0][0] + voronoi_graph[(0, i)][1][0]) / 2,
                                    (voronoi_graph[(0, i)][0][1] + voronoi_graph[(0, i)][1][1]) / 2])
                    vp.append((mid - points[i]) / (step * np.sqrt(np.sum(np.square(mid - points[i])))))
            else:
                if i == 1:
                    if defender_flag:
                        vp.append((points[0]-points[i]) / (step * np.sqrt(np.sum(np.square(points[0] - points[i])))))
                    else:
                        vp.append(uj_)
                else:
                    vp.append((points[0]-points[i]) / (step * np.sqrt(np.sum(np.square(points[0] - points[i])))))

        vp = np.array(vp)

        for i in range(1, 3):
            points[i] = points[i] + vp[i-1]


    # evader运动策略 随机运动 最大速度为1
    # Ve_x = E_[0][0] - points[0][0]
    # Ve_x = random.uniform(-0.2, 0.2)
    # Ve_y = random.uniform(E_[0][1], E_[1][1]) - points[0][1]
    # Ve_y = np.sqrt(0.04-np.square(Ve_x))
    if evader_strategy.empty():
        # 队列中不存在策略
        if points[0][0] == d[0] and points[0][1] == d[1]:
            # 到达上次设定的目的地
            D = [0, 0]
        else:
            # 未到达上次设定的目的地
            D = d / (step * np.sqrt(np.sum(np.square(d - points[0]))))
    else:
        # 队列中存在策略，获取策略，更新D值
        evader_ = evader_strategy.get()
        d = [evader_[0] - points[0][0], evader_[1] - points[0][1]]
        d = np.array(d)
        D = d / (step * np.sqrt(np.sum(np.square(d))))


    points[0] = points[0] + D

    for i in range(0, 3):
        if points[i][0] > bounding_box[1]:
            points[i][0] = bounding_box[1]

        if points[i][0] < bounding_box[0]:
            if points[i][1] >= E_[1][1] and points[i][1] <= E_[0][1]:
                points[i][0] = bounding_box[0]

        if points[i][1] > bounding_box[3]:
            points[i][1] = bounding_box[3]

        if points[i][1] < bounding_box[2]:
            points[i][1] = bounding_box[2]

        ax[i] = points[i][0]
        ay[i] = points[i][1]

    plt.pause(0.0001)
    plt.ioff
    plt.show()

if points[0][0] - D[0] < 0:
    print("Escape Successfully")
else:
    print("Capture Successfully!")

