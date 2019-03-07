# -- coding:utf-8 --
from paras import *
from create_examples import create_examples_with_prev_fluent
from tally import make_origin_data, check_result, find_path_return_travel_time, direction_to_english
import csv
#import matplotlib.pyplot as plt
from pyplotz.pyplotz import PyplotZ
from pyplotz.pyplotz import plt
import datetime
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
pltz = PyplotZ()
pltz.enable_chinese()

IMPROVED = False


def __tabulate(data, actions):
    """
    统计data（视频分片集）中每种actions组合出现的次数。
    :param data: create_examples得到的new_data，格式为 fluent | actions | prev_fluent
    :param actions: 列表，包含在data中，想要研究的actions（其实fluent也可以）的列号
    :return: 假设要研究k个actions，则返回一个2^k长度的列表，每一项依次表示按二进制序的actions组合出现的次数。
    例如：研究2个actions（A1,A2），返回[c00, c01, c10, c11]，其中cij表示data中“A1=i, A2=j”的行数，也即该actions组合出现的次数。
    """
    n_actions = len(actions)
    true_observations = [0] * (2 ** n_actions)
    for row in data:
        index = 0
        for i in range(n_actions-1, -1, -1):
            action_index = actions[i]
            col = n_actions - i - 1
            index += (2 ** col) * row[action_index]
        true_observations[index] += 1
    return true_observations


def calc_KL(all_f, all_h):
    if np.sum(all_f) > 1.001 or np.sum(all_f) < 0.999 or \
                    np.sum(all_h) > 1.001 or np.sum(all_h) < 0.999:
        print("all_f or all_h is out of bounds.")
        exit(233)
    all_info = 0.0
    for i in range(len(all_f)):
        if all_f[i] != 0:
            if all_h[i] == 0:
                print("h has some zeros where it should not!")
                exit(233)
            all_info += all_f[i] * np.log(all_f[i] / all_h[i])
    if all_info < -0.00001:
        print('we calculated a negative KL divergence')
        exit(233)
    return all_info


def pursuit(data, n_iterations=10, output_types=None, action_weights=None):
    """
    采用论文的迭代算法，逐步构建因果图。
    :param data: 【4-1】create_examples得到的new_data，格式为 fluent | actions | prev_fluent
    :param n_iterations: 最大迭代次数
    :param output_types: 一个数组，且为[0,1,2,3]的子集。
    0: F=0->0,  1: F=1->0,  2: F=0->1,  3: F=1->1
    :param action_weights: 一个列表，action_weight[1]对应data中第一个action对应的权重
    :return: 迭代结果：[best_output, best_actions, best_action_score, causal_effect]
    """
    if output_types is None:
        output_types = [0, 1, 2, 3]
    n_rows, n_cols = np.array(data).shape

    if action_weights is None:
        action_weights = [1] * (n_cols - 1)  # 第0位是填充对齐作用，并不代表weights

    table_of_info_gains = np.zeros((n_iterations, n_cols*4+10))  # TODO: 想办法利用它打印InfoGain表格？？
    best_actions = [0]  # 每轮迭代选出的最佳动作A_i
    best_output = [0]   # A_i对应的ΔF类别：0->0为1，1->0为2，0->1为3，1->1为4
    best_action_score = [0.0]    # A_i对应的 information gain
    causal_effect = [0.0]   # 神秘的评估打分，由概率式求出。大于0表示这条因果边较为可信
    # stored_h = np.zeros((8, n_cols))
    # stored_f = np.zeros((8, n_cols))
    #
    # # 【4-2-1】f & h 的初始化
    # tmp = __tabulate(data, [0, n_cols-1])   # 四种ΔF在data中的出现频次：[F:0->0, F:1->0, F:0->1, F:1->1]
    # h_fluent = [tmp[0], tmp[0], tmp[1], tmp[1], tmp[2], tmp[2], tmp[3], tmp[3]]  # P_ΔF（频次，尚未归一化为频率）
    # for action_index in range(1, n_cols-1):     # 对于每个动作A_i对应的下标i（data的第二列至倒数第二列为actions）
    #     # 初始化stored_f，也是真实观测的分布f
    #     stored_f[:, action_index] = __tabulate(data, [0, n_cols-1, action_index])   # 得到8个值：ΔF与A_i的联合分布
    #     stored_f[:, action_index] /= np.sum(stored_f[:, action_index])  # 将频次归一化为频率（概率的近似）
    #     # 初始化stored_h，也即模型的初始分布h。它等于 P_A(即h_action) * P_ΔF(即h_fluent)
    #     tmp = __tabulate(data, [action_index])  # [a0, a1]，即该动作A_i=0的出现频次，以及它=1的频次
    #     h_action = tmp * 4  # [a0, a1, a0, a1, a0, a1, a0, a1]
    #     stored_h[:, action_index] = np.multiply(h_fluent, h_action)  # 也得到8个值：ΔF的分布与A_i的分布之乘积
    #     stored_h[:, action_index] /= np.sum(stored_h[:, action_index])  # 将频次归一化为频率（概率的近似）
    # print("stored_f: \n", stored_f)
    # print("stored_h: \n", stored_h)     # OK
    # exit(2233)

    # 【4-2-1】f & h 的初始化
    # TODO: 尝试加权【README尝试3.】
    h_fluent = np.zeros(8)  # [0] * 8
    h_action = np.zeros((2, n_cols))  # [[0] * (n_cols - 2)] * 2
    stored_f = np.zeros((8, n_cols))  # [[0] * (n_cols - 2)] * 8
    stored_h = np.zeros((8, n_cols))  # [[0] * (n_cols - 2)] * 8
    fluent_type_map = [[0, 1], [2, 3]]
    for row in data:
        fluent_type = fluent_type_map[row[0]][row[-1]]
        # 更新 h_fluent
        h_fluent[fluent_type * 2] += 1
        h_fluent[fluent_type * 2 + 1] += 1
        # 计算这一视频片段有多少个=1的动作，分类讨论，以引入权重
        actions = np.array(row[1:-1])
        n_happened_actions = np.sum(actions)
        if n_happened_actions == 1:
            all_action_indices = set(range(n_cols - 1)) ^ {0}
            action_index = np.where(actions == 1)[0][0] + 1  # 加1得到在row里的下标，才和原来的action_index一致
            include_action_indices = {action_index}
            exclude_action_indices = list(all_action_indices ^ include_action_indices)

            # 更新 h_action
            h_action[1][action_index] += 1
            h_action[0][exclude_action_indices] += 1

            # 更新 stored_f
            fluent_indices = [fluent_type * 2, fluent_type * 2 + 1]
            stored_f[fluent_indices[1]][action_index] += 1
            stored_f[fluent_indices[0]][exclude_action_indices] += 1

        else:
            all_action_indices = set(range(n_cols - 1)) ^ {0}
            action_indices = np.where(actions == 1)[0]
            action_indices = [x + 1 for x in action_indices]
            exclude_action_indices = list(all_action_indices ^ set(action_indices))

            # 更新 h_action
            h_action[0][exclude_action_indices] += 1
            for action_index in action_indices:
                h_action[1][action_index] += action_weights[action_index]

            # 更新 stored_f
            fluent_indices = [fluent_type * 2, fluent_type * 2 + 1]
            stored_f[fluent_indices[0]][exclude_action_indices] += 1
            for action_index in action_indices:
                stored_f[fluent_indices[1]][action_index] += action_weights[action_index]
    h_action = np.array(list(h_action) * 4)
    for action_index in range(1, n_cols-1):
        stored_f[:, action_index] /= np.sum(stored_f[:, action_index])
        stored_h[:, action_index] = np.multiply(h_fluent, h_action[:, action_index])  # 也得到8个值：ΔF的分布与A_i的分布之乘积
        stored_h[:, action_index] /= np.sum(stored_h[:, action_index])  # 将频次归一化为频率（概率的近似）
    # print("stored_f: \n", stored_f)
    # print("stored_h: \n", stored_h)
    # exit(2233)

    # 【4-2-2】迭代算法，每轮迭代找出一条因果边
    for iteration in range(n_iterations):
        # 下一条因果边
        next_best_action = 0  # 的Action
        next_best_action_score = 0.0  # 的Info Gain
        next_best_output = 0  # 的ΔF
        next_best_f = []  # 的[f00, f01, f10, f11]，表示该因果边的真实分布的概率
        # 双重循环output_type（4种ΔF）、action_index（多种A_i），即遍历每种ΔF--A_i组合，找出信息增益最大的一组
        for output_type in output_types:    # 分别对应ΔF为0->0，1->0，0->1，1->1
            indices10 = 2 * output_type      # ΔF取该output_type，A=0：对应的stored_f(或stored_h)中的行号
            indices11 = 2 * output_type + 1  # ΔF取该output_type，A=1：……
            indices00 = list(set(range(0, 7, 2)) ^ {indices10})  # ΔF不取该output_type，A=0：……
            indices01 = list(set(range(1, 8, 2)) ^ {indices11})  # ΔF不取该output_type，A=1：……
            for action_index in range(1, n_cols-1):     # 对于每个动作A_i
                is_redundant = False
                for i in range(len(best_actions)):  # 查看因果图中是否已经有这条因果边了
                    if IMPROVED:
                        if best_actions[i] == action_index:  # TODO:【README：尝试2】
                            is_redundant = True  # 如果已经有了，则跳过这条边（不再重复计算）
                            break
                    else:
                        if best_actions[i] == action_index and best_output[i] == output_type:
                            is_redundant = True  # 如果已经有了，则跳过这条边（不再重复计算）
                            break
                if not is_redundant:  # 如果没有，则开始这条因果边的相关计算；
                    # 该动作Ai关于该output_type的真实分布f = [f00, f01, f10, f11]
                    f = stored_f[:, action_index]  # 长为8，该动作Ai对应的真实分布f（stored_f中的对应列）
                    sum00 = np.sum(f[[x for x in indices00]])
                    sum01 = np.sum(f[[x for x in indices01]])
                    f = [sum00, sum01, f[indices10], f[indices11]]  # 长为4，[f00, f01, f10, f11]，表示该因果边相关的真实分布的概率
                    f /= np.sum(f)  # 归一化
                    # 该动作Ai关于该output_type的当前模型分布h = [h00, h01, h10, h11]
                    h = stored_h[:, action_index]
                    sum00 = np.sum(h[[x for x in indices00]])
                    sum01 = np.sum(h[[x for x in indices01]])
                    h = [sum00, sum01, h[indices10], h[indices11]]  # 长为4，[h00, h01, h10, h11]，表示该因果边相关的模型分布的概率
                    h /= np.sum(h)
                    # 计算information gain
                    info = calc_KL(f, h)
                    # info = calc_KL(f, h) * action_weights[action_index]  # TODO:【README：尝试1，权重】
                    table_of_info_gains[iteration][(action_index-1)*4+output_type] = info
                    if info > next_best_action_score:
                        next_best_action = action_index
                        next_best_action_score = info
                        next_best_output = output_type
                        next_indices00 = indices00
                        next_indices01 = indices01
                        next_indices10 = indices10
                        next_indices11 = indices11
                        next_best_f = f  # [f00, f01, f10, f11]
        if next_best_action == 0:   # then no action was found
            print("End iterations: when no actions were found.")
            break
        # 【4-2-3】记录选中的最佳output_type——action组合（即因果边）及其Info Gain值（Info Gain最高）
        best_output.append(next_best_output)
        best_actions.append(next_best_action)
        best_action_score.append(next_best_action_score)

        # 【4-2-4】计算causal effect：P(ΔF | do(A)) - P(ΔF | do(not A))
        f = next_best_f  # 选中的因果边的真实分布的概率
        new_causal_effect = (f[3] / (f[3] + f[1]) - f[2] / (f[2] + f[0]))  # / (1 - f[2] / (f[2] + f[0]))
        # new_causal_effect = f[3] / (f[3] + f[1])
        causal_effect.append(new_causal_effect)

        # 迭代结束的threshold
        if next_best_action_score < .00001:
            print("End iterations: when IG < threshold.")
            break

        # 【4-2-5】将新的因果边加入模型，更新stored_h（对这条因果边，用真实分布的概率替代原有的模型分布概率）
        stored_h[next_indices11, next_best_action] = f[3]
        stored_h[next_indices10, next_best_action] = f[2]

        # 下面两行含义：例如output_type=0时，stored_h[2]在stored_h的第2,4,6行中所占的比重，乘以，真实概率f[0]=f00=“ΔF不为0->0且A_i=0时”的概率。即加权×f[0]
        tmp_sum = np.sum(stored_h[next_indices00, next_best_action])
        stored_h[next_indices00, next_best_action] *= f[0] / tmp_sum
        tmp_sum = np.sum(stored_h[next_indices01, next_best_action])
        stored_h[next_indices01, next_best_action] *= f[1] / tmp_sum

        stored_h[:, next_best_action] /= np.sum(stored_h[:, next_best_action])  # 归一化
        # print('iteration =', iteration)

    # OUTPUT
    output = [best_output, best_actions, best_action_score, causal_effect]
    for item in output:
        del item[0]
    return output


def __plot_output(result, title, save_file, best_output, best_actions):
    _, n_iterations = np.array(result).shape
    info = result[2]
    causal_effect = [round(x, 4) for x in result[3]]
    x = np.array(range(0, n_iterations))
    right_sub = [i for i in range(n_iterations) if causal_effect[i] > 0.1]
    wrong_sub = [i for i in range(n_iterations) if causal_effect[i] <= 0.1]
    pltz.set_figure_size(12, 10)
    pltz.plot(x, info)
    pltz.plot(x[right_sub], info[right_sub], 'go', linewidth=3, markersize=10)
    pltz.plot(x[wrong_sub], info[wrong_sub], 'rx', linewidth=3, markersize=10)
    # pltz.xlabel('Iteration Number')
    pltz.xticks([])
    pltz.ylabel('Information Gain')
    pltz.title(title)
    pltz.legend()

    col_labels = [str(i) for i in range(n_iterations)]
    row_labels = ['Selected A_i', 'Selected ΔF', 'TE']
    table_vals = np.array([best_actions, best_output, causal_effect])

    # row_colors = ['red', 'gold', 'green']

    my_table = plt.table(cellText=table_vals, colWidths=[1 / n_iterations] * n_iterations,
                         rowLabels=row_labels, colLabels=col_labels,
                         colLoc='center',
                         loc='bottom')
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(data_path + save_file)
    plt.show()
    # plt.close('all')


def __print_result(result, title, save_file, time_lag, action_lag, _action_names=None):
    if _action_names is None:
        print("action names is none!")
        return
    # 【4-3-1】将结果打印到屏幕
    _action_names = [''] + _action_names
    _action_names = np.array(_action_names)
    output_str = np.array(['0->0', '1->0', '0->1', '1->1'])
    best_output = output_str[result[0]]
    print("time_lag = "+str(time_lag)+", action_lag = "+str(action_lag))
    print("best output:")
    print(best_output, ', len =', len(result[0]))
    print("best actions:")
    best_actions = _action_names[result[1]]
    print(best_actions)
    np.set_printoptions(formatter={'all': lambda x: '%.4f' % x})
    result = np.array(result)
    print("best action score:")
    print(result[2])
    print("causal effect:")
    print(result[3])
    np.set_printoptions()   # 重置，避免写best_output到文件时出bug
    # 【4-3-2】画出迭代-InfoGain图
    __plot_output(result, title, save_file, best_output, best_actions)
    # 【4-3-3】将打印的结果存入文件，方便后续查看
    # with open(result_recorder_file, 'a') as fil:
    #     fil.write('======== ' + title + ' ========\n')
    #     fil.write(str(datetime.datetime.now())+'\n')    # 输出时间信息方便后续查看
    #     fil.write("time_lag = "+str(time_lag)+", action_lag = "+str(action_lag)+'\n')
    #     fil.write('best output: ')
    #     fil.write(str(best_output))
    #     fil.write('\nbest actions: ')
    #     fil.write(str(best_actions))
    #     fil.write('\nbest action score: ')
    #     np.set_printoptions(formatter={'all': lambda x: '%.4f' % float(x)})
    #     fil.write(str(result[2]))
    #     fil.write('\ncausal effect: ')
    #     fil.write(str(result[3]))
    #     fil.write('\n\n')
    #     np.set_printoptions()


def _debug_write_example_data_file():
    new_data = create_examples_with_prev_fluent()
    with open(example_data_file, 'w') as f:
        for i in range(len(new_data)):
            for j in range(len(new_data[0])):
                f.write('\t%d' % new_data[i][j])
            f.write('\n')


def _debug_learn_door_data():
    with open(door_data_file) as fil:
        reader = csv.reader(fil)
        data = list(reader)
        n_rows, n_cols = np.array(data).shape
        new_data = [[int(x[i]) for i in range(n_cols)] for x in data]
        print(np.array(new_data).shape)
        result = pursuit(new_data, 40)
        __print_result(result, 'door', 'door.png')


def learning(title, save_file, action_names, action_weights, time_lag=3, action_lag=2, intersect_bool=True, origin_data_number=0):
    # 【4-1】根据origin_data和分片标准*_lag、intersect_bool，对视频进行分片，得到Fluent | Actions | prev_Fluent形式的new_data
    new_data = create_examples_with_prev_fluent(time_lag, action_lag, intersect_bool, origin_data_number)
    print("example_data.shape =", np.array(new_data).shape)

    # 【4-2】执行迭代算法，得到因果图结果result
    # result = pursuit(new_data, 40, [0, 1, 2, 3])
    result = pursuit(new_data, 15, [0, 1, 2, 3], action_weights)

    # 【4-3】将结果可视化地打印出来
    __print_result(result, title, save_file, time_lag, action_lag, action_names)


def analyze():
    c_ssid = ['HK-173']
    e_ssid = 'HK-83'
    exclude_actions = []  # ['HK-95由南向北', 'HK-91由东向西']  # ['HK-173由西向东', 'HK-105由北向南', 'HK-381由南向北', 'HK-105由东向西']
    # c_thres = [23]    # TODO: 5. 【实验】调整参数
    # e_thres = 93
    c_thres = None
    e_thres = None
    intersect_bool = True
    e_mode = 'sum'

    # ****************** DATA PREPARATION ********************
    print("****************** DATA PREPARATION ********************")

    # 【1】生成origin_data（关键帧TimeTag--Fluent(0/1)--Actions(0/1)）文件
    c_thres, e_thres, action_names, e_directions = \
        make_origin_data(c_ssid, e_ssid, c_thres, e_thres, e_mode, exclude_actions)

    # 【2】求出两个路口间的旅行时间，确定分片依据time_lag
    travel_times = find_path_return_travel_time(c_ssid, e_ssid)
    action_weights_tmp = 1. / (np.array(travel_times) / np.sum(travel_times))
    action_weights = [0]
    for action_name in action_names:
        for k in range(len(c_ssid)):
            if c_ssid[k] in action_name:
                action_weights.append(action_weights_tmp[k])
    action_weights = np.array(action_weights) / np.min(action_weights[1:])
    np.set_printoptions(formatter={'all': lambda x: '%.4f' % x})
    print("Action_weights:", action_weights)
    if not IMPROVED:
        action_weights = np.ones(len(action_names)+1)
        print("Action_weights:", action_weights)
    np.set_printoptions()
    # exit(1111)
    time_delays = [int(x / 60 / 5) for x in travel_times]
    time_delay = int(np.max(time_delays))    # 取各路口 travel time 最大值来作为视频分片依据。
    # TODO: 9. 改进算法：计算RF时，考虑利用这个time_delays数组来加权，delay越短的因果作用越大，是否合理？
    time_lag = time_delay + 1
    action_lag = len(c_ssid)  # TODO: 7. 【实验】选取更多的action_lag，尝试intersect_bool = False
    print("time_lag =", time_lag, ", action_lag =", action_lag)


    # ****************** LEARNING ********************
    print("****************** LEARNING ********************")
    if e_mode == 'sum':
        # 【3】设定算法结果图的标题(title)、存储结果的文件名(save_file)
        print("+++++++++++++++++ " + e_ssid + " +++++++++++++++++")
        title = 'c=' + str(c_ssid) + ', e=' + e_ssid + '-' + \
                '\nc_thres=' + str(c_thres) + ', e_thres=' + str(e_thres)
        save_file = 'c=' + str(c_ssid) + ', e=' + e_ssid + '.png'

        # 【4】学习因果图
        learning(title, save_file, action_names, action_weights, time_lag, action_lag, intersect_bool, -1)

    elif e_mode == 'each':
        for j in range(0, len(e_directions)):  # 若为'each'模式，则对于每个果路口方向，分别学习因果图。
            e_direction = e_directions[j]
            print("+++++++++++++++++ "+e_ssid+'-'+e_direction+" +++++++++++++++++")
            title = 'c='+str(c_ssid)+', e='+e_ssid+'-'+direction_to_english(e_direction) + \
                    '\nc_thres='+str(c_thres)+', e_thres='+str(e_thres[j])
            save_file = 'c='+str(c_ssid)+', e='+e_ssid + '-' + str(j) + '.png'
            learning(title, save_file, action_names, time_lag, action_lag, intersect_bool, j)


    # check_result(c_ssid, e_ssid, time_delay, c_thres, e_thres)


if __name__ == '__main__':
    analyze()


