import pandas as pd
import scipy.stats as st
import numpy as np
import simpy
import time
import functools
from SimComponents_rev import Part, Sink, Process, Source, Monitor

start_run = time.time()
env = simpy.Environment()

process_list = ['Cutting', 'Bending', 'LineHeating']
part = [i+1 for i in range(22000)]
columns = pd.MultiIndex.from_product([[i for i in range(len(process_list)+1)], ['start_time', 'process_time', 'process']])
df = pd.DataFrame(columns=columns, index=part)
start_time = [0.112 * i for i in range(22000)]

for i in range(len(process_list) + 1):
    # Sink 모델링
    if i == len(process_list):
        df[(i, 'start_time')] = None
        df[(i, 'process_time')] = None
        df[(i, 'process')] = 'Sink'
    # 공정 모델링
    else:
        df[(i, 'start_time')] = 0 if i != 0 else start_time
        df[(i, 'process_time')] = None
        df[(i, 'process')] = process_list[i]

process_time_1 = st.norm.rvs(loc=0.5, size=100000)
df[(0, 'process_time')] = process_time_1[process_time_1 > 0][:22000]
process_time_2 = st.norm.rvs(loc=0.5, size=100000)
df[(1, 'process_time')] = process_time_2[process_time_2 > 0][:22000]
process_time_3 = st.norm.rvs(loc=3.0, size=100000)
df[(2, 'process_time')] = process_time_3[process_time_3 > 0][:22000]

parts = []
for i in range(len(df)):
    parts.append(Part(df.index[i], df.iloc[i]))

server_num = [7, 4, 8]
filepath = './result/event_log_ME.csv'
Monitor = Monitor(filepath)

each_MTTF = functools.partial(np.random.triangular, 432, 480, 528)
each_MTTR = functools.partial(np.random.triangular, 3.6, 4.0, 4.4)
delaying = functools.partial(np.random.triangular, 1/6, 1/4, 1/3)

MTTF = {}
MTTR = {}

MTTF['Cutting'] = [None for _ in range(7)]
MTTF['Bending'] = [each_MTTF for _ in range(4)]
MTTF['LineHeating'] = [each_MTTF, each_MTTF, None, None, None, None, None, None]

MTTR['Cutting'] = [None for _ in range(7)]
MTTR['Bending'] = [each_MTTR for _ in range(4)]
MTTR['LineHeating'] = [each_MTTR, each_MTTR, None, None, None, None, None, None]

model = {}
Source = Source(env, parts, model, Monitor)
priority = {}
priority['LineHeating'] = [1, 1, 2, 2, 2, 2, 2, 2]

model['Cutting'] = Process(env, 'Cutting', server_num[0], model, Monitor, delaying = delaying)
model['Bending'] = Process(env, 'Bending', server_num[1], model, Monitor, MTTR=MTTR, MTTF=MTTF, delaying=delaying)
model['LineHeating'] = Process(env, 'LineHeating', server_num[2], model, Monitor, routing_logic='priority', priority=priority, MTTR=MTTR, MTTF=MTTF, delaying=delaying)
model['Sink'] = Sink(env, Monitor)

# Simulation
start = time.time()  # 시뮬레이션 실행 시작 시각
env.run(until=2400)
finish = time.time()  # 시뮬레이션 실행 종료 시각

print('#' * 80)
print("Results of PBS simulation")
print('#' * 80)

# 코드 실행 시간
print("data pre-processing : ", start - start_run)
print("total time : ", finish - start_run)
print("simulation execution time :", finish - start)

print('#' * 80)
print("Makespan : ", model['Sink'].last_arrival)

event_tracer = Monitor.save_event_tracer()