import csv
from paras import *


def __find_by_time_indices(index, data, time, time_lag):
    """
    找到从当前行（第index行）往前数、固定时长（time_lag）之内的data行（关键帧数据）
    :param index: 
    :param data: origin_data数据
    :param time: 当前行的时间标签（第几帧）
    :param time_lag: 
    :return: 找出来的data行号的集合
    """
    by_time_indices = []
    for i in range(index - 1, 0, -1):
        time_i = data[i][0]
        if time - time_i > time_lag:
            break
        by_time_indices.append(i)
    return set(by_time_indices)


def __find_by_action_indices(index, data, action, action_lag):
    """
    找到从当前行（第index行）往前数、固定动作数（action_lag）之内的data行（关键帧数据）。
    这里第i-1行与第i行Actions字段不同，则认为发生了一次动作，动作数+1。
    :param index: 
    :param data: origin_data数据
    :param action: 当前行的动作字段（data[index][2:]）
    :param action_lag: 
    :return: 找出来的data行号的集合
    """
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
    """
    确保by_lag_indices里的几行fluent确实一致，如果不一致，则取保持一致的连续的那几行
    :param by_lag_indices: 
    :param data: 
    :param fluent: 
    :return: 
    """
    # 倒着往上找，是否有某一行的fluent与第index行（当前考察的example备选行）的fluent不一样，如果有则其行号存入fluent_fail
    fluent_fail = 0
    for tmp in sorted(by_lag_indices, reverse=True):
        if data[tmp][1] != fluent:
            fluent_fail = tmp
            break
    # 把行号小于fluent_fail的那些行（Fluent不一致）从by_lag_indices中舍去
    ret_by_lag_indices = []
    if len(by_lag_indices) != 0:
        for index in by_lag_indices:
            if index > fluent_fail:
                ret_by_lag_indices.append(index)
    return ret_by_lag_indices


def create_examples_with_prev_fluent(time_lag=3, action_lag=1, intersect_bool=True, origin_data_number=0):

    # 特别处理出错的情况
    if time_lag == 0 and action_lag == 0:
        print("ERROR: time_lag and action_lag both zero.")
        return

    # 获得origin_data的文件名
    filename = origin_data_file[:-4]+str(origin_data_number)+'.csv'
    if origin_data_number == -1:    # 代表e_mode为'sum'模式
        filename = origin_data_file

    with open(filename) as f:  # 读入origin_data
        reader = csv.reader(f)
        data = list(reader)[1:]
        n_rows, n_cols = np.array(data).shape
        data = [[int(x[i]) for i in range(n_cols)] for x in data]
        inertial_index = n_cols + 1
        n_actions = n_cols - 2

        new_data = []   # 目标结果
        example_indices = []    # fluent发生改变（ΔF=1）的data行号（关键帧行号）
        actions_in_examples = []    # 所有与fluent改变对应的example中，所有actions的data行号

        # 【4-1-1】获取fluent发生改变（ΔF=1）的example
        last_fluent = -1
        for index in range(1, n_rows):
            time = data[index][0]
            fluent = data[index][1]
            action = data[index][2:]
            example = [0] * inertial_index  # example为new_data中的一行（fluent | actions | prev_fluent）
            example[0] = time
            example[1] = fluent
            if index == 1:  # 第一行特殊处理
                last_fluent = fluent
            example[-1] = last_fluent
            if fluent != last_fluent:   # fluent发生改变，需构建example
                # 找到符合lag要求之内的关键帧行号：by_lag_indices
                by_time_indices = __find_by_time_indices(index, data, time, time_lag)
                by_action_indices = __find_by_action_indices(index, data, action, action_lag)
                if time_lag > 0 and action_lag > 0 and intersect_bool:
                    by_lag_indices = by_time_indices.intersection(by_action_indices)
                else:
                    by_lag_indices = by_time_indices.union(by_action_indices)
                # TODO: Error Checking codes...
                by_lag_indices = list(by_lag_indices.difference(
                    set(actions_in_examples)))  # 将by_lag_indices中与actions_in_examples重复的关键帧行号去掉
                example_indices.append(index)   # 当前行记为ΔF=1的example
                actions_in_examples.extend(by_lag_indices)  # 与此example相关的几行的行号，计入到actions_in_examples中，以免以后被重复计算
                # 将by_lag_indices这几行的action“或”起来，得到这个视频片段中出现过的所有动作，作为example的action字段。
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

        # 【4-1-2】获取fluent保持一致（ΔF=0）的example
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

                # 添加inertial_example后，将它对应的几行的动作全都抹去，即这几行不再算作example
                # TODO: 这样做是否合理？
                for row_index in by_lag_indices:
                    for action_col in range(2, n_cols):
                        data[row_index][action_col] = 0

        # with open(data_path + 'new_data.csv', 'w') as fn:
        #     fn.write("Time_Tag,F_Status,HK-173-由东向西,HK-173-由西向东, Prev_F_Status\n")
        #     for data_line in new_data:
        #         fn.write(str(data_line[0]))
        #         for i in range(1, len(data_line)):
        #             if i == 0:
        #                 continue
        #             fn.write(','+str(data_line[i]))
        #         fn.write('\n')
        # exit(233)

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

