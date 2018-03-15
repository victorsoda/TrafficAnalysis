import csv
import numpy as np


data_path = 'data/'
volume_file = data_path + 'volume.csv'
travel_time_file = data_path + 'travel_time.csv'


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
                ssid_dict[ssid] = np.zeros(20)
                ssid_dict[ssid][cdbh] = volume
        n_ssid = len(ssid_dict)
        print('Num of SSID =', n_ssid)
        ssid_list = sorted(ssid_dict.items(), key=lambda x: np.sum(x[1]))
        print('%5s %20s %10s %10s' % ('rank', 'SSID', 'daily vol', '5min avg'))
        cnt = 0
        for item in ssid_list:
            cnt += 1
            v_sum = np.sum(item[1])
            print('%5.2f %20s %10.0f %10.2f' % (float(cnt) / n_ssid, item[0], v_sum, v_sum / 288))


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

ssid_volume()
roadid_traveltime()

