##Traffic Analysis

####概要

数据集：宣城交通数据集（http://www.openits.cn/openData2/746.jhtml）

研究思路：

- 首先研究一条直线上的两个路口之间的因果影响情况，跑通实验，验证分析思路的正确性。
- 其次选取几个流量较小的路口作为可能的cause，选取一个流量很大的路口作为effect，探究其中的因果关系与权重。


主要方法：

- 主要参考论文：[A.Fire and S.-C. Zhu. "Learning perceptual causality from video." ACM Trans. Intell. Syst. Technol., 7(2):23:1–23:22, 2016.](http://amyfire.com/projects/learningcausality)
- 大体思路：逐步构建因果树模型，刻画交通路口间的因果关系。
- 构建模型的算法：真实数据有一个共现概率分布$$f$$，当前模型有一个共现概率分布$$h$$，模型构建的方法为每次增加一条信息增益（information gain）最大的因果边，使模型分布$$h$$以最大步长逼近真实分布$$f$$。




####tally.py：初步数据统计

- ssid_volume()：统计各路口各方向一天内的总流量、平均流量
- roadid_traveltime()：统计各路段的平均旅行时间、平均路程
- direction_volume()：统计要研究的路口各方向的车流量，每5分钟为一个时间段（可能需要plot绘直方图，辅助确定F、A的阈值的选取，继而定义F、A）
- make_origin_data()（暂时是取平均值作为流量阈值参数，或直接人为设定，**后应改为更加科学、统一的自动阈值设定**）：生成每一行为 “time | fluent | actions” 格式的 origin_data.csv，保证相邻两行的各对应值不完全相同
- find_path_return_travel_time()：输入目标路口id和“因”路口id，输出两者间的旅行时间。暂时使用了Dijkstra算法找最短路的时间，**后应考虑如“最短的三条路径的加权平均”等**




####learning.py：分析数据，学习因果关系

- create_examples_with_prev_fluent()（后单独写在一个.py文件中）：
  - 从time | fluent | actions 的origin_data到“关键帧”： fluent | actions | prev_fluent
- pursuit()：通过迭代，逐渐构建因果边，返回构建的因果树结果。



存在问题：

- 数据记录实时性不足：每5分钟才统计一次车流量、旅行时间，而大多数相邻路口间的车程不足5分钟。




####待研究问题

- causal_effect值的含义？**（与学长讨论）**
- 如果A、B路口间有***多条路径***，或者在一天中的不同时间，**它们之间的time delay（平均行车时间）是变化的**：
  - 问题1：难以确定分片的标准（可以取max？）
  - 问题2：如果算法发现A对B有影响，怎么确定A对**多少分钟后**的路口B有影响？
- ***多个路口***对一个路口的影响：
  - 代码拓展
  - 把RF的定义从平均变为**加权**，应该怎么写？
- 讨论




####论文要求

- 参考文献至少要有20来篇
- 图表要清晰
- 至少要50页
- ​





####实验记录：

见result_recorder.txt。









