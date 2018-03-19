import csv
import numpy as np
from paras import *


def __find_by_time_indices(index, data, time, time_lag):
    by_time_indices = []
    for i in range(index - 1, 1, -1):
        time_i = data[i][0]
        if time - time_i > time_lag:
            break
        by_time_indices.append(i)
    return set(by_time_indices)


def __find_by_action_indices(index, data, action, action_lag):
    by_action_indices = []
    cnt = 0
    last_action = action
    for i in range(index - 1, 1, -1):
        action_i = data[i][2:]
        if action_i == last_action:
            continue
        cnt += 1
        if cnt > action_lag:
            break
        by_action_indices.append(i)
        last_action = action_i
    return set(by_action_indices)


def create_examples_with_prev_fluent(time_lag=2, action_lag=1, intersect_bool=True):

    if time_lag == 0 and action_lag == 0:
        print("ERROR: time_lag and action_lag both zero.")
        return

    with open(origin_data_file) as f:
        reader = csv.reader(f)
        z = [int(x) for x in list(reader)[1:]]
        print(z)

        exit(233)

        data = np.array([1, 1, 1])
        n_rows, n_cols = data.shape
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
            example = np.zeros(inertial_index)
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
                tmp_action = np.zeros(n_actions)
                if len(by_lag_indices) != 0:
                    for tmp_index in by_lag_indices:
                        tmp_action = tmp_action | data[tmp_index][2:]
                example[2:-1] = tmp_action
                new_data.append(example[1:])
            last_fluent = fluent

        return new_data


new_data = create_examples_with_prev_fluent()

print(new_data)
