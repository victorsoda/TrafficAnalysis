import csv
from paras import *

n_daily_time_tag = 288     # 按5分钟划分一天的时间，可分为288个时段
n_lanes_max = 20    # 最多不超过20个车道


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
            # print('%10d %15.2f %15.2f' % (item[0], item[1]['tt'], item[1]['dist']))
            item[1]['dist'] /= item[1]['cnt']
            item[1]['tt'] /= item[1]['cnt']

        n_road = len(road_dict)
        print('Num of road =', n_road)
        road_list = sorted(road_dict.items(), key=lambda x: x[1]['tt'])
        print('%5s %10s %15s %15s' % ('rank', 'ROAD_ID', 'travel_time', 'dist'))
        cnt = 0.0
        for item in road_list:
            cnt += 1
            print('%5.2f %10d %15.2f %15.2f' % (float(cnt)/n_road, item[0], item[1]['tt'], item[1]['dist']))


def __time_2_tag(starttime):
    hms = starttime.split(' ')
    if len(hms) == 1:   # starttime = "2016/12/15"，表示凌晨0点
        return 0
    h, m, s = hms[1].split(':')
    return int(int(h) * 12 + int(m) / 5)


def direction_volume(ssid, to_print='all'):
    """
    统计目标路口每5分钟内各方向的车流量。
    :param ssid: 目标路口
    :param to_print: 打印选项，'each'：各方向分别打印，'sum'：只打印各方向总和，'all'：上述都打印，'none'：都不打印
    :return: dir_vol_dict：各方向的车流量序列（字典格式）, vol_sum：各方向车流量总和的序列（列表格式）
    """
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
            print("Volume Sequence: ", vol_seq)
            print("Max_vol: %5.2f%5sMin_vol: %5.2f%5sAvg_vol: %5.2f%5sMedian_vol: %5.2f" %
                  (np.max(vol_seq), '', np.min(vol_seq), '', np.mean(vol_seq[excluder:]), '', np.median(vol_seq[excluder:])))
    if to_print == 'all' or to_print == 'sum':
        print("========== 各方向总计 ==========")
        print("Volume Sequence: ", vol_sum)
        print("Max_vol: %5.2f%5sMin_vol: %5.2f%5sAvg_vol: %5.2f%5sMedian_vol: %5.2f" %
              (np.max(vol_sum), '', np.min(vol_sum), '', np.mean(vol_sum[excluder:]), '', np.median(vol_sum[excluder:])))
    return dir_vol_dict, vol_sum


def make_origin_data():

    c_ssid = "HK-173"   # cause_ssid
    e_ssid = "HK-92"    # effect_ssid

    c_dv_dict, _ = direction_volume(c_ssid, 'each')  # cause_direction_volume_dict
    __, e_sum = direction_volume(e_ssid, 'sum')  # effect_volume_sum

    c_thres = 30    # TODO: 调整参数
    e_thres = 120

    origin_data = []
    last_line = []
    for time_tag in range(n_daily_time_tag):
        line = list()
        line.append(time_tag)
        line.append(int(e_sum[time_tag] > e_thres))
        for direction in c_dv_dict.keys():
            line.append(int(c_dv_dict[direction][time_tag] > c_thres))
        if time_tag == 0 or line[1:] != last_line[1:]:
            origin_data.append(line)
        last_line = line

    with open(origin_data_file, 'w') as f:
        header = 'Time_Tag,F_Status'
        for direction in c_dv_dict.keys():
            header += ',' + c_ssid + '-' + direction
        header += '\n'
        f.write(header)
        for line in origin_data:
            f.write(str(line[0]))
            for item in line[1:]:
                f.write(',' + str(item))
            f.write('\n')



# ssid_volume()
# roadid_traveltime()
# print(__time_2_tag("2016/12/15 1:20:00"))

make_origin_data()




