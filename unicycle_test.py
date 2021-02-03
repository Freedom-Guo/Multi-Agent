import numpy as np
import matplotlib.pyplot as plt
import math
import queue

agent_num = 5
evader_num = 2
pursuer_num = 3
delta = 3

V_pm = [1, 1, 1]
W_pm = [1, 1, 1]
pursuer = [0, 0, 0]

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


# 绘制图的坐标X和Y
ax = []
ay = []
# 动态图
plt.ion()
fig = plt.figure()

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
Rc = np.zeros((evader_num, pursuer_num))

for i in range(evader_num):
    for j in range(pursuer_num):
        Rc[i][j] = np.sqrt(np.sum(np.square(points[i] - points[j + evader_num])))


def is_all_capture(distance):
    global death, evader_num, pursuer_num
    index = 0
    print(distance)
    for i in range(evader_num):
        if is_capture(distance[i][:]):
            if i not in death:
                death.append(i)
                for j in range(pursuer_num):
                    distance[i][j] = 0.0
    for i in range(evader_num):
        if i in death:
            index += 1
    return (index != evader_num)


def is_capture(distance):
    global agent_num, pursuer_num
    for j in range(pursuer_num):
        #print("distance:")
        #print(distance)
        if distance[j] <= 1:
            return True
    return False


def get_Vpx(index, e, p, delta_):
    global evader_num, pursuer_num
    v = np.array(p[index])
    temp2 = 0
    temp3 = 0
    temp4 = 0
    for i_ in range(evader_num - len(death)):
        # E1-ENe
        v1 = np.array(e[i_])
        temp1 = 0
        for j_ in range(pursuer_num):
            # P1-PNp
            v2 = np.array(p[j_])
            temp1 = temp1 + np.linalg.norm(v1 - v2) ** ((-1) * delta_)
        temp2 = temp2 + pursuer_num / temp1
        temp3 = (v[0] - v1[0]) * (np.linalg.norm(v1 - v) ** ((-1) * delta_ - 2))
        temp4 = temp4 + temp3 / (temp1 ** 2)
    return pursuer_num * (temp2 ** (1 / delta_ - 1)) * temp4


def get_Vpy(index, e, p, delta_):
    global evader_num, pursuer_num
    v = np.array(p[index])
    temp2 = 0
    temp3 = 0
    temp4 = 0
    for i_ in range(evader_num - len(death)):
        # E1-ENe
        v1 = np.array(e[i_])
        temp1 = 0
        for j_ in range(pursuer_num):
            # P1-PNp
            v2 = np.array(p[j_])
            temp1 = temp1 + np.linalg.norm(v1 - v2) ** ((-1) * delta_)
        temp2 = temp2 + pursuer_num / temp1
        temp3 = (v[1]- v1[1]) * (np.linalg.norm(v1 - v) ** ((-1) * delta_ - 2))
        temp4 = temp4 + temp3 / (temp1 ** 2)
    return pursuer_num * (temp2 ** (1 / delta_ - 1)) * temp4


# 获取鼠标键入，和下方鼠标键入evader策略同时使用
d = [points[0][0], points[0][1]]
cid = fig.canvas.mpl_connect('button_press_event', evader_controller)

D = [0, 0]

while is_all_capture(Rc):
    af_points = []
    for i in range(agent_num):
        if i not in death:
            af_points.append(points[i])

    af_points = np.array(af_points)

    # 绘制pursuers和evader的点
    plt.clf()
    for i in range(evader_num):
        if i in death:
            plt.plot(ax[i], ay[i], 'kx', markersize=5)
        else:
            plt.plot(ax[i], ay[i], 'gp', markersize=5)
    plt.plot(ax[evader_num:], ay[evader_num:], 'ro', markersize=5)

    # 绘制边界
    plt.plot([bounding_box[0], bounding_box[0]], [bounding_box[2], bounding_box[3]], 'b-')
    plt.plot([bounding_box[1], bounding_box[1]], [bounding_box[2], bounding_box[3]], 'b-')
    plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[2], bounding_box[2]], 'b-')
    plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[3], bounding_box[3]], 'b-')

    V_px = []
    V_py = []
    V_p = []
    W_p = []
    theta_pr = []
    for i in range(pursuer_num):
        V_px.append(get_Vpx(i, af_points[:(evader_num - len(death))], af_points[(evader_num - len(death)):], delta))
        V_py.append(get_Vpy(i, af_points[:(evader_num - len(death))], af_points[(evader_num - len(death)):], delta))
        theta_pr.append((1 / 2) * np.pi - math.atan(V_px[i] / V_py[i]))
        V_p.append((-1) * V_pm[i] * np.sign(V_px[i] * math.cos(pursuer[i]) + V_py[i] * math.sin(pursuer[i])))
        W_p.append((-1) * (W_pm[i]) * np.sign(pursuer[i] - theta_pr[i]))
        af_points[i + evader_num - len(death)] = [af_points[i + evader_num - len(death)][0] + V_p[i] * math.cos(pursuer[i]),
                                              af_points[i + evader_num - len(death)][1] + V_p[i] * math.sin(pursuer[i])]
        pursuer[i] = pursuer[i] + W_p[i]

    count = 0

    for i in range(evader_num):
        if i == 0:
            if i not in death:
                Ve_x = 0
                Ve_y = 1
                D = [Ve_x, Ve_y]
                D = np.array(D)
                af_points[count] = af_points[count] + D
                count += 1
        elif i == 1:
            if i not in death:
                Ve_x = 1
                Ve_y = 0
                D = [Ve_x, Ve_y]
                D = np.array(D)
                af_points[count] = af_points[count] + D
                count += 1
        # if i == 0:
        #     if i not in death:
        #         if evader_strategy.empty():
        #             if points[0][0] == d[0] and points[0][1] == d[1]:
        #                 D = [0, 0]
        #             else:
        #                 D = D
        #         else:
        #             evader_ = evader_strategy.get()
        #             d = [evader_[0], evader_[1]]
        #             vec = [d[0] - points[0][0], d[1] - points[0][1]]
        #             vec = np.array(vec)
        #             D = 0.9 * vec / np.sqrt((np.sum(np.square(vec))))
        #         af_points[count] = af_points[i] + D
        #         count += 1
        # else:
        #     if i not in death:
        #         Ve_x = np.random.uniform(-0.9, 0.9)
        #         Ve_x = np.random.uniform(-0.9, 0.9)
        #         Ve_y = np.sqrt(0.81 - Ve_x ** 2) * np.random.choice([-1, 1])
        #         D = [Ve_x, Ve_y]
        #         D = np.array(D)
        #         af_points[count] = af_points[i] + D
        #         count += 1

    af_index = 0
    be_points = []
    for i in range(agent_num):
        if i in death:
            be_points.append(points[i])
        else:
            be_points.append(af_points[af_index])
            af_index += 1

    points = np.array(be_points)

    for i in range(agent_num):
        ax[i] = points[i][0]
        ay[i] = points[i][1]

    for i in range(evader_num):
        for j in range(agent_num - evader_num):
            Rc[i][j] = np.sqrt(np.sum(np.square(points[i] - points[j + evader_num])))

    plt.pause(0.01)
    plt.ioff

plt.clf()
plt.axis('equal')
plt.axis('off')
plt.plot(ax[:evader_num], ay[:evader_num], 'kx', markersize=5)
plt.plot(ax[evader_num:], ay[evader_num:], 'ro', markersize=5)
# 绘制边界
plt.plot([bounding_box[0], bounding_box[0]], [bounding_box[2], bounding_box[3]], 'b-')
plt.plot([bounding_box[1], bounding_box[1]], [bounding_box[2], bounding_box[3]], 'b-')
plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[2], bounding_box[2]], 'b-')
plt.plot([bounding_box[0], bounding_box[1]], [bounding_box[3], bounding_box[3]], 'b-')
plt.pause(3)
plt.ioff
print("Capture Successfully!")





