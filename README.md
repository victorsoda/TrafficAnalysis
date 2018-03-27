##Traffic Analysis

数据集：宣城交通数据集（http://www.openits.cn/openData2/746.jhtml）



tally.py：初步数据统计

- 统计各路口各方向一天内的总流量、平均流量
- 统计各路段的平均旅行时间、平均路程
- 统计要研究的路口各方向的车流量，每5分钟为一个时间段（可能需要作图，辅助确定F、A的阈值选取）



研究思路：

- 首先研究一条直线上的两个路口之间的因果影响情况，跑通实验，验证分析思路的正确性。
- 其次选取几个流量较小的路口作为可能的cause，选取一个流量很大的路口作为effect，探究其中的因果关系与权重。




learning.py：分析数据，学习因果关系

- create_examples_with_prev_fluent()：从time | fluent | actions 的origin_data到“关键帧”： fluent | actions | prev_fluent
- pursuit()：通过迭代，逐渐构建因果边



存在问题：

- 数据记录实时性不足：每5分钟才统计一次车流量、旅行时间，而大多数相邻路口间的车程不足5分钟。

