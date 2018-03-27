import pprint
import numpy as np

data_path = 'data/'
volume_file = data_path + 'volume.csv'
travel_time_file = data_path + 'travel_time.csv'
lane_direction_file = data_path + 'lane_direction.csv'
origin_data_file = data_path + 'origin_data.csv'    # 生成的初始数据文件
door_data_file = data_path + 'door_data.csv'    # 使用源代码开门视频的create_examples输出的data，验证pursuit的正确性
example_data_file = data_path + 'example_data.txt'  # create_examples生成的文件
result_recorder_file = data_path + 'result_recorder.txt'


pp = pprint.PrettyPrinter()

# a = np.array([[1, 3, 7], [2, 2, 3]])
# b = np.mean(a)
# print(b, type(b))

# b = [0, 1]
# print(np.sum(b))
# print(a[[x for x in b]])
