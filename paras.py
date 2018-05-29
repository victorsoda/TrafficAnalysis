import pprint
import numpy as np

data_path = 'data/'
volume_file = data_path + 'volume.csv'
travel_time_file = data_path + 'travel_time.csv'
road_id_travel_time_file = data_path + 'road_id_travel_time.csv'
ssid2_road_id_file = data_path + 'ssidA&B_road_id.csv'
lane_direction_file = data_path + 'lane_direction.csv'
origin_data_file = data_path + 'origin_data.csv'    # 生成的初始数据文件
door_data_file = data_path + 'door_data.csv'    # 使用源代码开门视频的create_examples输出的data，验证pursuit的正确性
example_data_file = data_path + 'example_data.txt'  # create_examples生成的文件
result_recorder_file = data_path + 'result_recorder.txt'
TIME_SLICE = 5   # 宣城数据集的时间分片为5分钟
INF = 1e10

pp = pprint.PrettyPrinter()
# print(origin_data_file[:-4])

# a = np.zeros((8, 2))
# print(a)

# row = np.array([0, 0, 1, 1, 0])
# print(list(row) * 4)
# # ind = [2, 3]
# a = {2, 3}
# [x for x in ] += 1
# print(row)
# action_index = np.where(np.array(row) == 1)
# action_index = [x + 1 for x in action_index[0]]
# print(action_index)

# t_val = np.array([[11, 12, 13, 14, 15, 16, 17],
#          [21, 22, 23, 24, 25, 26, 27],
#          [31, 32, 33, 34, 35, 36, 37]])
# print(t_val[:, 0:3])
# print(t_val[:, 3:])


