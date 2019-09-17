from matplotlib import pyplot as plt
import setting
import numpy as np


def draw_learning_curve_raw(episodes, scores, file_path, title):
    #  对最后100个数据取平均值
    plt.plot(episodes, scores)
    label = ['score']
    plt.legend(label, loc='upper left')
    plt.title(title)
    plt.show()


def draw_learning_curve_ave100(episodes, scores, file_path, title):
    s = []
    for i in range(len(score)):
        if i < 100:
            s.append(np.average(score[:i + 1]))
        else:
            s.append(np.average(score[i - 100:i + 1]))
    plt.plot(episodes, s)
    label = ['average score of past 100 step']
    plt.legend(label, loc='upper left')
    plt.title(title)
    plt.show()


# 将状态转化为q_table里的index
def idx_state(env, state):
    # 认为env只有20个
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / setting.one_feature

    # 将连续的env转化为离散的
    positioone_feature = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])

    # 通过离散的env和
    state_idx = positioone_feature + velocity_idx * setting.one_feature
    return state_idx


if __name__ == '__main__':
    score = []
    for i in range(200):
        score.append(i)
    print(score)
    draw_learning_curve_raw(score, score, "", "le")
