##Traffic Analysis

####概要

数据集：宣城交通数据集（http://www.openits.cn/openData2/746.jhtml）

研究思路：

- 首先研究一条直线上的两个路口之间的因果影响情况，跑通实验，验证分析思路的正确性。
- 其次选取几个流量较小的路口作为可能的cause，选取一个流量很大的路口作为effect，探究其中的因果关系与权重。



####tally.py：初步数据统计

- ssid_volume()：统计各路口各方向一天内的总流量、平均流量
- roadid_traveltime()：统计各路段的平均旅行时间、平均路程
- direction_volume()：统计要研究的路口各方向的车流量，每5分钟为一个时间段（可能需要plot绘直方图，辅助确定F、A的阈值的选取，继而定义F、A）
- make_origin_data()（暂时是写死路口、流量阈值参数，后应改进为自动生成）：生成每一行为 “time | fluent | actions” 格式的 origin_data.csv，保证相邻两行的各对应值不完全相同




####learning.py：分析数据，学习因果关系

- create_examples_with_prev_fluent()（后单独写在一个.py文件中）：
  - 从time | fluent | actions 的origin_data到“关键帧”： fluent | actions | prev_fluent
- pursuit()：通过迭代，逐渐构建因果边，返回构建的因果树结果。



存在问题：

- 数据记录实时性不足：每5分钟才统计一次车流量、旅行时间，而大多数相邻路口间的车程不足5分钟。






####实验记录：

#####HK-173对HK-92的影响









