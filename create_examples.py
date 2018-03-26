import csv
from paras import *


def __find_by_time_indices(index, data, time, time_lag):
    by_time_indices = []
    for i in range(index - 1, 0, -1):
        time_i = data[i][0]
        if time - time_i > time_lag:
            break
        by_time_indices.append(i)
    return set(by_time_indices)


def __find_by_action_indices(index, data, action, action_lag):
    by_action_indices = []
    cnt = 0
    last_action = action
    for i in range(index - 1, 0, -1):
        action_i = data[i][2:]
        if action_i == last_action:
            continue
        cnt += 1
        if cnt > action_lag:
            break
        by_action_indices.append(i)
        last_action = action_i
    return set(by_action_indices)


def __ensure_consistency(by_lag_indices, data, fluent):
    fluent_fail = 0
    for tmp in sorted(by_lag_indices, reverse=True):
        if data[tmp][1] != fluent:
            fluent_fail = tmp
            break
    ret_by_lag_indices = []
    if len(by_lag_indices) != 0:
        for index in by_lag_indices:
            if index > fluent_fail:
                ret_by_lag_indices.append(index)
    return ret_by_lag_indices


def create_examples_with_prev_fluent(time_lag=3, action_lag=1, intersect_bool=True):

    if time_lag == 0 and action_lag == 0:
        print("ERROR: time_lag and action_lag both zero.")
        return

    with open(origin_data_file) as f:
        reader = csv.reader(f)
        data = list(reader)[1:]
        n_rows, n_cols = np.array(data).shape
        data = [[int(x[i]) for i in range(n_cols)] for x in data]
        inertial_index = n_cols + 1
        n_actions = n_cols - 2

        new_data = []   # 目标结果
        example_indices = []    # fluent发生改变的data行号
        actions_in_examples = []    # 所有与fluent改变对应的example中，所有actions的data行号

        last_fluent = -1
        for index in range(1, n_rows):
            time = data[index][0]
            fluent = data[index][1]
            action = data[index][2:]
            example = [0] * inertial_index
            example[0] = time
            example[1] = fluent
            if index == 1:
                last_fluent = fluent
            example[-1] = last_fluent
            if fluent != last_fluent:   # fluent发生改变，需构建example
                by_time_indices = __find_by_time_indices(index, data, time, time_lag)
                by_action_indices = __find_by_action_indices(index, data, action, action_lag)
                if time_lag > 0 and action_lag > 0 and intersect_bool:
                    by_lag_indices = by_time_indices.intersection(by_action_indices)
                else:
                    by_lag_indices = by_time_indices.union(by_action_indices)
                # TODO: Error Checking codes...
                by_lag_indices = list(by_lag_indices.difference(set(actions_in_examples)))
                example_indices.append(index)
                actions_in_examples.extend(by_lag_indices)
                tmp_action = [0] * n_actions
                if len(by_lag_indices) != 0:
                    for tmp_index in by_lag_indices:
                        for i in range(len(tmp_action)):
                            tmp_action[i] = tmp_action[i] | data[tmp_index][2 + i]
                example[2:-1] = tmp_action
                new_data.append(example)
            last_fluent = fluent

        # print([data[item][0] for item in example_indices])
        # print([data[item][0] for item in actions_in_examples])
        # 保证已经算过的有action的行不再重复算
        for row_index in actions_in_examples:
            for action_col in range(2, n_cols):
                data[row_index][action_col] = 0

        # 获取fluent保持一致的example
        for index in range(n_rows-1, 0, -1):
            # 有动作，则记录为一个example
            if any(data[index][2:]) \
                    and index not in example_indices:  # TODO:（之前记录过example的时间点不再记录？）
                time = data[index][0]
                fluent = data[index][1]
                action = data[index][2:]

                by_time_indices = __find_by_time_indices(index, data, time, time_lag)
                by_action_indices = __find_by_action_indices(index, data, action, action_lag)
                if time_lag > 0 and action_lag > 0 and intersect_bool:
                    by_lag_indices = by_time_indices.intersection(by_action_indices)
                else:
                    by_lag_indices = by_time_indices.union(by_action_indices)
                by_lag_indices = __ensure_consistency(by_lag_indices, data, fluent)
                tmp_action = [0] * n_actions
                if len(by_lag_indices) != 0:
                    for tmp_index in by_lag_indices:
                        for i in range(len(tmp_action)):
                            tmp_action[i] = tmp_action[i] | data[tmp_index][2 + i]
                inertial_example = [time, fluent]
                inertial_example.extend(tmp_action)
                inertial_example.append(fluent)
                new_data.append(inertial_example)

                for row_index in by_lag_indices:
                    for action_col in range(2, n_cols):
                        data[row_index][action_col] = 0

        for item in new_data:
            del item[0]
        return new_data


# new_data = create_examples_with_prev_fluent()
#
# pp.pprint(new_data)
# print(np.array(new_data).shape)













# data = [[0, 0, 0, 0],
#         [80, 0, 1, 0],
#         [81, 0, 0, 0],
#         [82, 1, 0, 0],  # index = 3
#         [83, 1, 1, 0],
#         [88, 0, 1, 0]]
# index = 5
# time = 88
# time_lag = 3
# action_lag = 1
# action = data[index][2:]
# # print(__find_by_time_indices(index, data, time, time_lag))
# # print(__find_by_action_indices(index, data, action, action_lag))
# print(data[index][2:])

