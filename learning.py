from paras import *
import csv
from create_examples import create_examples_with_prev_fluent


def __tabulate(data, actions):  # 统计data（视频分片集）中每种actions组合出现的次数
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


def pursuit(data, n_iterations=20, output_types=None):
    if output_types is None:
        output_types = [1, 2]
    n_rows, n_cols = np.array(data).shape
    table_of_info_gains = np.zeros((n_iterations, n_cols*4+10))
    best_actions = [1]  # 每轮迭代选出的最佳动作A_i
    best_output = [0]   # A_i对应的ΔF类别，0->0为1，1->0为2，0->1为3，1->1为4
    best_action_score = [0.0]    # A_i对应的 information gain
    causal_effect = [0.0]
    stored_h = np.zeros((8, n_cols))
    stored_f = np.zeros((8, n_cols))
    # INITIALIZE OF f & h
    tmp = __tabulate(data, [0, n_cols-1])   # 四种ΔF的统计数量
    h_fluent = [tmp[0], tmp[0], tmp[1], tmp[1], tmp[2], tmp[2], tmp[3], tmp[3]]
    for action_index in range(1, n_cols-1):     # data的第二列至倒数第二列为actions
        # 初始化stored_f，也是真实观测的f
        stored_f[:, action_index] = __tabulate(data, [0, n_cols-1, action_index])
        stored_f[:, action_index] /= np.sum(stored_f[:, action_index])
        # 初始化stored_h = P_A * P_ΔF
        tmp = __tabulate(data, [action_index])
        h_action = tmp * 4
        stored_h[:, action_index] = np.multiply(h_fluent, h_action)
        stored_h[:, action_index] /= np.sum(stored_h[:, action_index])
    # print("stored_f: \n", stored_f)
    # print("stored_h: \n", stored_h)     # OK
    # PURSUIT LOOP
    for iteration in range(n_iterations):
        print('iteration =', iteration)
        next_best_action = 0
        next_best_action_score = 0.0
        next_best_output = 0
        next_best_f = []
        for output_type in output_types:    # 分别对应ΔF为0->0，1->0，0->1，1->1
            indices10 = 2 * output_type
            indices11 = 2 * output_type + 1
            indices00 = list(set(range(0, 7, 2)) ^ {indices10})
            indices01 = list(set(range(1, 8, 2)) ^ {indices11})
            for action_index in range(1, n_cols-1):     # 检查每个action和每个output type
                # TODO: make sure we don't repeat an action/output type combo
                is_redundant = False
                for i in range(len(best_actions)):
                    if best_actions[i] == action_index and best_output[i] == output_type:
                        is_redundant = True
                if not is_redundant:
                    # 该动作对应的真实分布f
                    f = stored_f[:, action_index]
                    sum00 = np.sum(f[[x for x in indices00]])
                    sum01 = np.sum(f[[x for x in indices01]])
                    f = [sum00, sum01, f[indices10], f[indices11]]
                    f /= np.sum(f)
                    # 当前模型的分布
                    h = stored_h[:, action_index]
                    sum00 = np.sum(h[[x for x in indices00]])
                    sum01 = np.sum(h[[x for x in indices01]])
                    h = [sum00, sum01, h[indices10], h[indices11]]
                    h /= np.sum(h)
                    # 计算information gain
                    info = calc_KL(f, h)
                    table_of_info_gains[iteration][(action_index-1)*4+output_type] = info
                    if info > next_best_action_score:
                        next_best_action = action_index
                        next_best_action_score = info
                        next_best_output = output_type
                        next_indices00 = indices00
                        next_indices01 = indices01
                        next_indices10 = indices10
                        next_indices11 = indices11
                        next_best_f = f
        if next_best_action == 0:   # then no action was found
            break
        # 记录选中的最佳action（info gain最高）
        best_output.append(next_best_output)
        best_actions.append(next_best_action)
        best_action_score.append(next_best_action_score)
        # 计算causal effect：P(ΔF | do(A)) - P(ΔF | do(not A))
        f = next_best_f
        new_causal_effect = (f[3] / (f[3] + f[1]) - f[2] / (f[2] + f[0])) / (1 - f[2] / (f[2] + f[0]))
        causal_effect.append(new_causal_effect)
        # 迭代结束的threshold
        if next_best_action_score < .00001:
            break
        # 将新的因果边加入模型，更新stored_h
        stored_h[next_indices11, next_best_action] = f[3]
        stored_h[next_indices10, next_best_action] = f[2]
        tmp_sum = np.sum(stored_h[next_indices00, next_best_action])
        stored_h[next_indices00, next_best_action] *= f[0] / tmp_sum
        tmp_sum = np.sum(stored_h[next_indices01, next_best_action])
        stored_h[next_indices01, next_best_action] *= f[1] / tmp_sum
        stored_h[:, next_best_action] /= np.sum(stored_h[:, next_best_action])
    # OUTPUT
    output = [best_output, best_actions, best_action_score, causal_effect]
    for item in output:
        del item[0]

    return output    # TODO

# new_data = create_examples_with_prev_fluent()
# with open(example_data_file, 'w') as f:
#     for i in range(len(new_data)):
#         for j in range(len(new_data[0])):
#             f.write('\t%d' % new_data[i][j])
#         f.write('\n')


with open(door_data_file) as fil:
    reader = csv.reader(fil)
    data = list(reader)
    n_rows, n_cols = np.array(data).shape
    new_data = [[int(x[i]) for i in range(n_cols)] for x in data]

    print(np.array(new_data).shape)
    result = pursuit(new_data, 40)
    print("best output:")
    print(result[0], ', len = ', len(result[0]))
    print("best actions:")
    print(result[1])
    print("best action score:")
    print(result[2])
    print("causal effect:")
    print(result[3])

# data = [[1, 1, 1, 1],
#         [1, 0, 0, 0],
#         [0, 0, 0, 0]]
# print(__tabulate(data, [0, 3]))

