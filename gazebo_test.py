import math
import numpy as np
import pygazebo as gazebo
import matplotlib.pyplot as plt

gazebo.initialize()
world = gazebo.new_world_from_file("/home/freedomguo/3D_collisionavoidance/world/turtlebot3_stage_2.world")

agents = world.get_agents()
agents[0].get_joint_names()
agents[1].get_joint_names()
agents[2].get_joint_names()
agents[3].get_joint_names()
agents[4].get_joint_names()

world.info()

evader_num = 2
pursuer_num = 3

evader = (((0, 2, 0), (0, 0, 0)), ((1, 1, 0), (0, 0, 0)))
pursuer = (((0, -0.4, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0)), ((0.4, 0, 0), (0, 0, 0)))
print("evader:")
print(evader)
print("pursuer:")
print(pursuer)
agents[0].set_pose(((0, -0.4, 0), (0, 0, 0)))
agents[1].set_pose(((0, 0, 0), (0, 0, 0)))
agents[2].set_pose(((0.4, 0, 0), (0, 0, 0)))
agents[3].set_pose(((0, 2, 0), (0, 0, 0)))
agents[4].set_pose(((1, 1, 0), (0, 0, 0)))

evader_flag = []
pursuer_flag = []
for i in range(evader_num):
    evader_flag.append(False)

for j in range(pursuer_num):
    pursuer_flag.append(False)

capture_radius = 0.1
ts = 1
count = 0
delta = 3

V_em = [0.25, 0.25]
V_pm = [0.26, 0.26, 0.26]
W_em = [1.82, 1.82]
W_pm = [0.8, 0.8, 0.8]

obs = agents[0].get_camera_observation("default::camera::camera_link::camera")
npdata3 = np.array(obs, copy=False)
plt.imshow(npdata3)
plt.savefig("frame"+str(count)+".png")


def is_all_escape(flag):
    for i in range(evader_num):
        if flag[i] is False:
            return True
    return False


def get_Vex(index, e, p, delta_):
    global evader_num, pursuer_num
    flag = False
    v = np.array([e[index][0][0], e[index][0][1]])
    temp2 = 0
    temp3 = 0
    temp4 = 0
    for i_ in range(evader_num):
        v1 = np.array([e[i_][0][0], e[i_][0][1]])
        temp1 = 0
        for j_ in range(pursuer_num):
            v2 = np.array([p[j_][0][0], p[j_][0][1]])
            temp1 = temp1 + np.linalg.norm(v1 - v2) ** ((-1) * delta_)
            if flag is False:
                temp3 = temp3 + (e[index][0][0] - p[j_][0][0]) * np.linalg.norm(v - v2) ** ((-1) * delta_ - 2)
                temp4 = temp4 + np.linalg.norm(v - v2) ** ((-1) * delta_)
                flag = True
        temp2 = temp2 + pursuer_num / temp1
    return pursuer_num * (temp2 ** (1 / delta_ - 1)) * (temp3 / (temp4 ** 2))


def get_Vey(index, e, p, delta_):
    global evader_num, pursuer_num
    flag = False
    v = np.array([e[index][0][0], e[index][0][1]])
    temp2 = 0
    temp3 = 0
    temp4 = 0
    for i_ in range(evader_num):
        v1 = np.array([e[i_][0][0], e[i_][0][1]])
        temp1 = 0
        for j_ in range(pursuer_num):
            v2 = np.array([p[j_][0][0], p[j_][0][1]])
            temp1 = temp1 + np.linalg.norm(v1 - v2) ** ((-1) * delta_)
            if flag is False:
                temp3 = temp3 + (e[index][0][1] - p[j_][0][1]) * np.linalg.norm(v - v2) ** ((-1) * delta_ - 2)
                temp4 = temp4 + np.linalg.norm(v - v2) ** ((-1) * delta_)
                flag = True
        temp2 = temp2 + pursuer_num / temp1
    return pursuer_num * (temp2 ** (1 / delta_ - 1)) * (temp3 / (temp4 ** 2))


def get_Vpx(index, e, p, delta_):
    global evader_num, pursuer_num
    v = np.array([p[index][0][0], p[index][0][1]])
    temp2 = 0
    temp3 = 0
    temp4 = 0
    for i_ in range(evader_num):
        # E1-ENe
        v1 = np.array([e[i_][0][0], e[i_][0][1]])
        temp1 = 0
        for j_ in range(pursuer_num):
            # P1-PNp
            v2 = np.array([p[j_][0][0], p[j_][0][1]])
            temp1 = temp1 + np.linalg.norm(v1 - v2) ** ((-1) * delta_)
        temp2 = temp2 + pursuer_num / temp1
        temp3 = (p[index][0][0] - e[i_][0][0]) * (np.linalg.norm(v1 - v) ** ((-1) * delta_ - 2))
        temp4 = temp4 + temp3 / (temp1 ** 2)
    return pursuer_num * (temp2 ** (1 / delta_ - 1)) * temp4


def get_Vpy(index, e, p, delta_):
    global evader_num, pursuer_num
    v = np.array([p[index][0][0], p[index][0][1]])
    temp2 = 0
    temp3 = 0
    temp4 = 0
    for i_ in range(evader_num):
        # E1-ENe
        v1 = np.array([e[i_][0][0], e[i_][0][1]])
        temp1 = 0
        for j_ in range(pursuer_num):
            # P1-PNp
            v2 = np.array([p[j_][0][0], p[j_][0][1]])
            temp1 = temp1 + np.linalg.norm(v1 - v2) ** ((-1) * delta_)
        temp2 = temp2 + pursuer_num / temp1
        temp3 = (p[index][0][1] - e[i_][0][1]) * (np.linalg.norm(v1 - v) ** ((-1) * delta_ - 2))
        temp4 = temp4 + temp3 / (temp1 ** 2)
    return pursuer_num * (temp2 ** (1 / delta_ - 1)) * temp4


while is_all_escape(evader_flag):
    V_ex = []
    V_ey = []
    V_px = []
    V_py = []
    V_e = []
    V_p = []
    W_e = []
    W_p = []
    pos_e = []
    pos_p = []
    theta_er = []
    theta_pr = []
    count += 1

    for i in range(evader_num):
        pos_e.append([evader[i][0][0], evader[i][0][1]])
        V_ex.append(get_Vex(i, evader, pursuer, delta))
        V_ey.append(get_Vey(i, evader, pursuer, delta))
        theta_er.append((1 / 2) * np.pi - math.atan(V_ex[i] / V_ey[i]))
        V_e.append(V_em[i] * np.sign(V_ex[i] * math.cos(evader[i][1][2]) + V_ey[i] * math.sin(evader[i][1][2])))
        W_e.append((-1) * (W_em[i]) * np.sign(evader[i][1][2] - theta_er[i]))
    pos_e = np.array(pos_e)

    for j in range(pursuer_num):
        pos_p.append([pursuer[j][0][0], pursuer[j][0][0]])
        V_px.append(get_Vpx(j, evader, pursuer, delta))
        V_py.append(get_Vpy(j, evader, pursuer, delta))
        theta_pr.append((1 / 2) * np.pi - math.atan(V_px[j] / V_py[j]))
        V_p.append((-1) * V_pm[j] * np.sign(V_px[j] * math.cos(pursuer[j][1][2]) + V_py[j] * math.sin(pursuer[j][1][2])))
        W_p.append((-1) * (W_pm[j]) * np.sign(pursuer[j][1][2] - theta_pr[j]))
    pos_p = np.array(pos_p)

    for i in range(evader_num):
        for j in range(pursuer_num):
            if np.linalg.norm(pos_e[i] - pos_p[j]) < capture_radius:
                pursuer_flag[j] = True
                evader_flag[i] = True

    for i in range(evader_num):
        if evader_flag[i] is True:
            agents[i+pursuer_num].set_twist(((0, 0, 0), (0, 0, 0)))
        else:
            agents[i+pursuer_num].set_twist(((V_e[i]*math.cos(W_e[i])*ts, V_e[i]*math.sin(W_e[i])*ts, 0), (0, 0, W_e[i]*ts)))
        print(agents[i+pursuer_num].get_twist())
        print(((V_e[i]*math.cos(W_e[i])*ts, V_e[i]*math.sin(W_e[i])*ts, 0), (0, 0, W_e[i]*ts)))

    for j in range(pursuer_num):
        if pursuer_flag[j] is True:
            agents[j].set_twist(((0, 0, 0), (0, 0, 0)))
        else:
            agents[j].set_twist(((V_p[j]*math.cos(W_p[j]*ts), V_p[j]*math.sin(W_p[j])*ts, 0), (0, 0, W_p[j]*ts)))
        print(agents[j].get_twist())
        print(((V_p[j]*math.cos(W_p[j]*ts), V_p[j]*math.sin(W_p[j])*ts, 0), (0, 0, W_p[j]*ts)))

    world.step(20000)

    evader = []
    pursuer = []
    for i in range(evader_num):
        # print(agents[i+pursuer_num].get_pose())
        evader.append(agents[i+pursuer_num].get_pose())

    for j in range(pursuer_num):
        pursuer.append(agents[j].get_pose())

    pursuer = tuple(pursuer)
    evader = tuple(evader)

    obs = agents[0].get_camera_observation("default::camera::camera_link::camera")
    npdata3 = np.array(obs, copy=False)
    plt.imshow(npdata3)
    plt.savefig("frame"+str(count)+".png")
    # plt.show()
    print("evader:")
    print(evader)
    print("pursuer:")
    print(pursuer)

