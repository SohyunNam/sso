import pandas as pd
import scipy.stats as st
import numpy as np
import simpy
import time
import functools
import matplotlib.pyplot as plt

workforces = 2
blocks = 50000
SIM_TIME = 50000
process_list = ['Process1', 'Process2', 'Process3']
output = {}

for i in range(2, workforces+1):
    from SimComponents_rev import Part, Sink, Process, Source, Monitor, Resource
    env = simpy.Environment()
    # df_part: part_id
    part = [x for x in range(blocks)]
    columns = pd.MultiIndex.from_product(
        [[y for y in range(len(process_list) + 1)], ["start_time", "process_time", "process"]])
    data = pd.DataFrame([], columns=columns, index=part)
    IAT = st.expon.rvs(1, size=blocks)
    start_time = IAT.cumsum()

    # Process1
    data[(0, 'start_time')] = start_time
    data[(0, 'process_time')] = None
    data[(0, 'process')] = "Process1"

    data[(1, 'start_time')] = None
    data[(1, 'process_time')] = None
    data[(1, 'process')] = "Process2"

    data[(2, 'start_time')] = None
    data[(2, 'process_time')] = None
    data[(2, 'process')] = "Process3"

    # Sink
    data[(3, 'start_time')] = None
    data[(3, 'process_time')] = None
    data[(3, 'process')] = 'Sink'

    parts = []
    for x in range(len(data)):
        parts.append(Part(data.index[x], data.iloc[x]))

    service_time = functools.partial(np.random.exponential, 1)

    model = {}
    process_time = {}
    for name in process_list:
        process_time[name] = [service_time]

    wf_info = {}
    for j in range(i):
        wf_info["WF_{0}".format(j)] = {"skill": 1.0}

    workforce = {}
    for name in process_list:
        workforce[name] = [True]

    server_num = [1 for _ in range(len(process_list))]
    filepath = './result/eventlog_simple_ME_wf_{0}.csv'.format(i)
    Monitor = Monitor(filepath)
    Resource = Resource(env, model, Monitor, wf_info=wf_info)

    Source = Source(env, parts, model, Monitor)

    # process modeling
    for k in range(len(process_list) + 1):
        if k == len(process_list):
            model['Sink'] = Sink(env, Monitor)
        else:
            model[process_list[k]] = Process(env, process_list[k], server_num[k], model, Monitor, resource=Resource, process_time=process_time, workforce=workforce)

    env.run(until=SIM_TIME)
    output[i] = model['Sink'].parts_rec
    for name in process_list:
        print("{0} : {1}".format(name, model[name].machine[0].total_working_time/model[name].parts_sent))
    event_tracer = Monitor.save_event_tracer()
    event_tracer.to_csv(filepath)

print(output)

lists = sorted(output.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.show()
