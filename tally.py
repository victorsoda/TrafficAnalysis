# -- coding:utf-8 --
import csv
from paras import *
import matplotlib.pyplot as plt

n_daily_time_tag = 288     # 按5分钟划分一天的时间，可分为288个时段
n_lanes_max = 20    # 最多不超过20个车道
volume_cleaner = 0     # 前50个时段的车流量一般很少，可以被数据清洗掉


def ssid_volume():
    with open(volume_file) as f:
        reader = csv.reader(f)
        v_dat = list(reader)[1:]
        ssid_dict = {}
        for row in v_dat:
            ssid = row[0]
            cdbh = int(row[1])
            volume = int(row[-1])
            if ssid in ssid_dict:
                ssid_dict[ssid][cdbh] += volume
            else:
                ssid_dict[ssid] = np.zeros(n_lanes_max)
                ssid_dict[ssid][cdbh] = volume
        n_ssid = len(ssid_dict)
        print('Num of SSID =', n_ssid)
        ssid_list = sorted(ssid_dict.items(), key=lambda x: np.sum(x[1]))
        print('%5s %20s %10s %10s' % ('rank', 'SSID', 'daily vol', '5min avg'))
        cnt = 0
        for item in ssid_list:
            cnt += 1
            v_sum = np.sum(item[1])
            print('%5.2f %20s %10.0f %10.2f' % (float(cnt) / n_ssid, item[0], v_sum, v_sum / n_daily_time_tag))


def roadid_traveltime():
    with open(travel_time_file) as f:
        reader = csv.reader(f)
        tt_dat = list(reader)[1:]
        road_dict = {}
        for row in tt_dat:
            road_id = int(row[-3])
            tt = float(row[-4])
            speed = float(row[-5])
            dist = tt * speed
            if road_id in road_dict:
                road_dict[road_id]['dist'] += dist
                road_dict[road_id]['tt'] += tt
                road_dict[road_id]['cnt'] += 1
            else:
                road_dict[road_id] = {}
                road_dict[road_id]['dist'] = dist
                road_dict[road_id]['tt'] = tt
                road_dict[road_id]['cnt'] = 1
        for item in road_dict.items():    # cal average tt and dist
            item[1]['dist'] /= item[1]['cnt']
            item[1]['tt'] /= item[1]['cnt']

        n_road = len(road_dict)
        print('Num of roads =', n_road)
        road_list = sorted(road_dict.items(), key=lambda x: x[1]['tt'])
        print('%5s %10s %15s %15s' % ('rank', 'ROAD_ID', 'travel_time', 'dist'))
        cnt = 0.0
        for item in road_list:
            cnt += 1
            print('%5.2f %10d %15.2f %15.2f' % (float(cnt)/n_road, item[0], item[1]['tt'], item[1]['dist']))
    with open(road_id_travel_time_file, 'w') as f:
        header = ['rank', 'ROAD_ID', 'travel_time(minutes)', 'dist(?)']
        for i in range(len(header)-1):
            f.write(header[i] + ',')
        f.write(header[len(header)-1]+'\n')
        cnt = 0.0
        for item in road_list:
            cnt += 1
            f.write('%.2f,%d,%.2f,%.2f\n' % (float(cnt)/n_road, item[0], item[1]['tt'], item[1]['dist']))


# TODO: 10. 输入c_ssid、e_ssid，输出它们之间的time delay（平均旅行时间），需考虑多条路径的问题（暂时还是返回了最短路径的用时）
def find_path_return_travel_time(c_ssids, e_ssid):
    """
    输入目标路口id和“因”路口id，输出两者间的旅行时间
    :param c_ssids: “因”路口id（可能有多个）
    :param e_ssid: 目标路口id 
    :return: 旅行时间（秒）
    """
    with open(road_id_travel_time_file) as f:
        reader = csv.reader(f)
        dat = list(reader)[1:]
        road_tt_dict = {}
        for line in dat:
            road_tt_dict[line[1]] = float(line[2])
    with open(ssid2_road_id_file) as f:
        reader = csv.reader(f)
        dat = list(reader)[1:]
        distances = {}
        for line in dat:
            node1 = line[1]
            node2 = line[2]
            road_id_str = line[0]
            road_ids = road_id_str.split('+')
            tt = 0.0
            for road_id in road_ids:
                if '*' in road_id:
                    tmp = road_id.split('*')
                    if tmp[1] in road_tt_dict.keys():
                        tt += float(tmp[0]) * road_tt_dict[tmp[1]]
                    else:
                        # print("Travel_time data of ROAD_ID "+tmp[1]+" doesn't exist.")
                        tt = INF
                elif road_id in road_tt_dict.keys():
                    tt += road_tt_dict[road_id]
                else:
                    # print("Travel_time data of ROAD_ID " + road_id + " doesn't exist.")
                    tt = INF
            if tt >= INF:   # 旅行时间数据不存在的路段，寻路算法将不予考虑
                continue
            if node1 not in distances.keys():
                distances[node1] = {node2: tt}
            else:
                distances[node1][node2] = tt
            if node2 not in distances.keys():
                distances[node2] = {node1: tt}
            else:
                distances[node2][node1] = tt
        nodes = distances.keys()
        unvisited = {node: None for node in nodes}  # 把None作为无穷大使用
        visited = {}  # 用来记录已经松弛过的数组
        current = e_ssid  # 要找目标点e_ssid到其他点的距离
        current_distance = 0
        unvisited[current] = current_distance  # e_ssid到自己的距离记为0

        while True:
            for neighbour, distance in distances[current].items():
                if neighbour not in unvisited:
                    continue  # 被访问过了，跳出本次循环
                new_distance = current_distance + distance  # 新的距离
                if unvisited[neighbour] is None or unvisited[neighbour] > new_distance:  # 如果两个点之间的距离之前是无穷大或者新距离小于原来的距离
                    unvisited[neighbour] = new_distance  # 更新距离
            visited[current] = current_distance  # 这个点已经松弛过，记录
            del unvisited[current]  # 从未访问过的字典中将这个点删除
            if not unvisited:
                break  # 如果所有点都松弛过，跳出此次循环
            candidates = [node for node in unvisited.items() if node[1]]  # 找出目前还有拿些点未松弛过
            current, current_distance = sorted(candidates, key=lambda x: x[1])[0]  # 找出目前可以用来松弛的点

        # print(visited)
        ret = []
        for c_ssid in c_ssids:
            tt = visited[c_ssid]
            ret.append(tt)
            print("Time delay between "+c_ssid+" and "+e_ssid+" is "+str(tt)+"s ("+str(int(tt/60/5))+" * 5min)")
        return ret


def __time_2_tag(starttime):
    """
    将原始数据中诸如"2016/12/15 0:15:00"的时间数据映射到每5min划分的时段（time_tag）里（如3）
    :param starttime: 如"2016/12/15 0:15:00"
    :return: 如 3
    """
    hms = starttime.split(' ')
    if len(hms) == 1:   # starttime = "2016/12/15"，表示凌晨0点，对应tag=0
        return 0
    h, m, s = hms[1].split(':')
    return int(int(h) * 12 + int(m) / 5)


def direction_to_english(direction):
    if direction == u'由西向东':
        return 'West->East'
    if direction == u'由东向西':
        return 'East->West'
    if direction == u'由南向北':
        return 'North->South'
    if direction == u'由北向南':
        return 'South->North'
    return 'Exception!!!'


def __plot_vol_seq(dv_dict, ssid):
    x = np.array(range(0, n_daily_time_tag))
    colors = ['g', 'r', 'b', 'y', 'w']
    cnt = 0
    for direction in dv_dict.keys():
        color = colors[cnt]
        y = dv_dict[direction]
        plt.plot(x, y, color, linewidth=1, markersize=3, label=direction_to_english(direction))
        cnt += 1
    plt.xlabel('Time Tag of Day')
    plt.ylabel('Volume')
    plt.title(ssid)
    plt.legend()
    plt.show()


def direction_volume(ssid, to_print='none', data_cleaning=True):
    """
    统计目标路口每5分钟内各方向的车流量。
    :param ssid: 目标路口
    :param to_print: 打印选项，'each'：各方向分别打印，'sum'：只打印各方向总和，'all'：上述都打印，'none'：都不打印
    :return: dir_vol_dict：各方向的车流量序列（字典格式）, vol_sum：各方向车流量总和的序列（列表格式）
    """
    if to_print != 'none':
        print("**************** SSID = " + ssid + " ****************")
    dir_lane_dict = {}  # { "由东向西": [1, 2, 3, 4（车道编号）] }
    with open(lane_direction_file) as f:
        reader = csv.reader(f)
        ld_dat = list(reader)[2:]
        for row in ld_dat:
            if row[0] == ssid:
                direction = row[2]
                lane = int(row[1])
                if direction in dir_lane_dict:
                    dir_lane_dict[direction].append(lane)
                else:
                    dir_lane_dict[direction] = [lane]

    lane_dir_list = [''] * n_lanes_max  # ['', '由东向西'（下标1对应1号车道）, '由东向西', ... ] （方便反向查询）
    for key in dir_lane_dict.keys():
        for lane in dir_lane_dict[key]:
            lane_dir_list[lane] = key

    dir_vol_dict = {}   # { "由东向西": [0, 0, 22, 37, 0, 14, ...（下标0对应着0:00~0:05的车流量）] }
    for key in dir_lane_dict.keys():
        dir_vol_dict[key] = np.zeros(n_daily_time_tag)

    with open(volume_file) as f:
        reader = csv.reader(f)
        v_dat = list(reader)[1:]
        for row in v_dat:
            if row[0] == ssid:
                lane = int(row[1])
                direction = lane_dir_list[lane]
                vol = int(row[-1])
                start_time = row[2]
                time_tag = __time_2_tag(start_time)
                dir_vol_dict[direction][time_tag] += vol

    vol_sum = np.zeros(n_daily_time_tag)
    excluder = 0    # 用于排除凌晨车流很小的时段
    for direction in dir_vol_dict.keys():
        vol_seq = dir_vol_dict[direction]  # volume sequence
        vol_sum += vol_seq
        if to_print == 'all' or to_print == 'each':
            print("========== " + direction + " ==========")
            print("Volume Sequence: ")
            print(vol_seq)
            print("Max_vol: %5.2f%5sMin_vol: %5.2f%5sAvg_vol: %5.2f%5sMedian_vol: %5.2f" %
                  (np.max(vol_seq), '', np.min(vol_seq), '', np.mean(vol_seq[excluder:]), '', np.median(vol_seq[excluder:])))
    if to_print == 'all' or to_print == 'sum':
        print("========== 各方向总计 ==========")
        print("Volume Sequence: ", vol_sum)
        print("Max_vol: %5.2f%5sMin_vol: %5.2f%5sAvg_vol: %5.2f%5sMedian_vol: %5.2f" %
              (np.max(vol_sum), '', np.min(vol_sum), '', np.mean(vol_sum[excluder:]), '', np.median(vol_sum[excluder:])))
    if to_print != 'none':
        __plot_vol_seq(dir_vol_dict, ssid)  # 绘制一天流量折线图
    if data_cleaning:
        dv_dict = {}
        for key in dir_vol_dict.keys():
            dv_dict[key] = dir_vol_dict[key][volume_cleaner:]
        v_sum = vol_sum[volume_cleaner:]
        return dv_dict, v_sum
    else:
        return dir_vol_dict, vol_sum


def __write_origin_data(origin_data, action_names, filename):
    with open(filename, 'w') as f:
        header = 'Time_Tag,F_Status'
        for i in range(len(action_names)):
            header += ',' + action_names[i]
        header += '\n'
        f.write(header)
        for line in origin_data:
            f.write(str(line[0]))
            for item in line[1:]:
                f.write(',' + str(item))
            f.write('\n')


def make_origin_data(c_ssids, e_ssid, c_thres=None, e_thres=None, e_mode='sum', exclude_actions=None):
    """
    生成每一行为 “time | fluent | actions” 格式的 origin_data.csv，保证相邻两行的各对应值不完全相同
    :param c_ssids: [list] cause_ssid，作为“因”的路口id（可能有多个）
    :param e_ssid: effect_ssid，作为“果”的路口id
    :param c_thres: “因”路口单一方向车流量的阈值，大于c_thres则Action=1，否则为0（可能有多个）
    :param e_thres: “果”路口总车流量的阈值，大于e_thres则Fluent=1，否则为0 
    :param e_mode: 'sum'表示Fluent考虑“果”路口总流量，'each'表示考虑“果”路口各方向的流量
    :param exclude_actions: 动作名字的列表，表示不予考虑的动作。如["HK-173-由西向东"]
    :return: 
    """

    # 统计因路口、果路口各方向的车流量
    c_dv_dicts = []
    for c_ssid in c_ssids:
        c_dv_dict, _ = direction_volume(c_ssid)  # cause_direction_volume_dict
        c_dv_dicts.append(c_dv_dict)
    e_dv_dict, e_sum = direction_volume(e_ssid)  # effect_volume_sum
    e_directions = list(e_dv_dict.keys())
    assert len(c_ssids) == len(c_dv_dicts)

    # 若未人为设定，则计算出因、果的0-1阈值（暂定为平均值）
    if c_thres is None:
        c_thres = []
        for c_dv_dict in c_dv_dicts:
            dv = []
            for value in c_dv_dict.values():
                dv.append(value)
            c_thres.append(np.int64(np.mean(np.array(dv)) + 0.5))  # 平均数作为阈值，四舍五入
            # c_thres.append(np.int64(np.median(np.array(dv))))   # 中位数作为阈值
    if e_thres is None:
        if e_mode == 'sum':
            e_thres = np.int64(np.mean(np.array(e_sum)) + 0.5)
            # e_thres = np.int64(np.median(np.array(e_sum)))
        elif e_mode == 'each':
            e_thres = []
            for direction in e_directions:
                c_thres.append(np.int64(np.mean(np.array(e_dv_dict[direction])) + 0.5))
                # e_thres.append(np.int64(np.median(np.array(e_dv_dict[direction]))))
    print('c_thres =', c_thres, '\te_thres =', e_thres)

    if exclude_actions is None:
        exclude_actions = []

    # 获取Action（A1、A2...）的名称，以便之后输出的Action能看懂
    action_names = []
    for i in range(len(c_ssids)):
        for direction in c_dv_dicts[i].keys():
            action_name = c_ssids[i] + direction
            if action_name not in exclude_actions:
                action_names.append(action_name)
    print(action_names)

    # 生成origin_data文件
    if e_mode == 'sum':
        origin_data = []
        # last_line = []
        for time_tag in range(n_daily_time_tag - volume_cleaner):  # TODO:早高峰，晚高峰，多种时间段模式
            line = list()
            line.append(time_tag + volume_cleaner)  # 第0列：Time_Tag
            line.append(int(e_sum[time_tag] > e_thres))  # 第1列：Fluent值
            for i in range(len(c_dv_dicts)):
                c_dv_dict = c_dv_dicts[i]
                for direction in c_dv_dict.keys():
                    action_name = c_ssids[i] + direction
                    if action_name not in exclude_actions:
                        line.append(int(c_dv_dict[direction][time_tag] > c_thres[i]))  # 第2~n列：Action值
            # if time_tag == 0 or line[1:] != last_line[1:]:  # 如果当前行与上一行有区别
            origin_data.append(line)  # 则视作一个关键帧，放入origin_data中【改进：保留所有帧】
            # last_line = line
        # 将结果写到文件中
        filename = origin_data_file[:-4] + '.csv'
        __write_origin_data(origin_data, action_names, filename)

    elif e_mode == 'each':
        for j in range(len(e_thres)):  # 若为'each'模式，则对于每个果路口方向，单独生成一个origin_data_N文件
            e_direction = e_directions[j]
            origin_data = []
            last_line = []
            for time_tag in range(n_daily_time_tag - volume_cleaner):  # 数据清洗：凌晨的数据不具代表性，舍去
                line = list()
                line.append(time_tag + volume_cleaner)  # 第0列：Time_Tag
                line.append(int(e_dv_dict[e_direction][time_tag] > e_thres[j]))  # 第1列：Fluent值
                for i in range(len(c_dv_dicts)):
                    c_dv_dict = c_dv_dicts[i]
                    for direction in c_dv_dict.keys():
                        action_name = c_ssids[i] + direction
                        if action_name not in exclude_actions:
                            line.append(int(c_dv_dict[direction][time_tag] > c_thres[i]))  # 第2~n列：Action值
                if time_tag == 0 or line[1:] != last_line[1:]:  # 如果当前行与上一行有区别
                    origin_data.append(line)  # 则视作一个关键帧，放入origin_data中
                last_line = line
            # 将结果写到文件中
            filename = origin_data_file[:-4] + str(j) + '.csv'
            __write_origin_data(origin_data, action_names, filename)

    return c_thres, e_thres, action_names, e_directions


def check_result(c_ssid, e_ssid, time_delay, c_thres=None, e_thres=None):
    c_dv_dict, _ = direction_volume(c_ssid)  # cause_direction_volume_dict
    __, e_sum = direction_volume(e_ssid)  # effect_volume_sum
    if c_thres is None:
        dv = []
        for value in c_dv_dict.values():
            dv.append(value)
        c_thres = np.int64(np.mean(np.array(dv)) + 0.5)  # 四舍五入
    if e_thres is None:
        e_thres = np.int64(np.mean(np.array(e_sum)) + 0.5)
    print('c_thres =', c_thres, '\te_thres =', e_thres, '\tcausal_time_delay =', time_delay)

    print('Effect_sum_volume > e_thres: %.2f%%' %
          (np.float64(np.sum(np.array(e_sum) > e_thres)) * 100.0 / len(e_sum)))

    for direction in c_dv_dict.keys():
        delta_f_cnt = [0] * 4   # 0: 0->0, 1: 1->0, 2: 0->1, 3: 1->1
        delta_f = ["0->0", "1->0", "0->1", "1->1"]
        for i in range(len(e_sum)):
            if i < time_delay or i == len(e_sum) - 1:
                continue
            cause_time = i - time_delay
            if c_dv_dict[direction][cause_time] > c_thres:
                output_type = int(e_sum[i] > e_thres) + 2 * int(e_sum[i+1] > e_thres)
                delta_f_cnt[output_type] += 1
        print(direction+" 导致（"+str(TIME_SLICE*time_delay)+"分钟后）的目标路口流量：")
        delta_f_cnt = np.array(delta_f_cnt) / np.sum(delta_f_cnt)
        for i in range(len(delta_f_cnt)):
            print(delta_f[i]+": %.2f%%" % (delta_f_cnt[i] * 100))


# ssid_volume()
# roadid_traveltime()
# print(__time_2_tag("2016/12/15 1:20:00"))
# find_path_return_travel_time("HK-173", "HK-83")
# direction_volume('HK-145', to_print='all')

