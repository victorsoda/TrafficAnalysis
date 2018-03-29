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
TIME_SLICE = 5   # 宣城数据集的时间分片为5分钟


pp = pprint.PrettyPrinter()

# a = ['2', '3', '4']
# with open('1.txt', 'w') as f:
#     f.write(str(a))
