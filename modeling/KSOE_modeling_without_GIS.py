import pandas as pd
import numpy as np
import simpy
import random
import time
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from SimComponents_Layout import Part, Source, Process, Sink, Monitor, Assembly

start_running = time.time()

# input data
data_all = pd.read_excel('../data/Layout_Activity_A0001.xlsx')
block_assembly = pd.read_excel('../data/Layout_BOM_A0001.xlsx')

''' ## Activity data pre-processing ## '''
''' 
convert integer to datetime
and conver datetime to integer for timeout function and calculating process time
'''
data_all['시작일'] = pd.to_datetime(data_all['시작일'], format='%Y%m%d')
data_all['종료일'] = pd.to_datetime(data_all['종료일'], format='%Y%m%d')
initial_date = data_all['시작일'].min()

data_all['시작일'] = data_all['시작일'].apply(lambda x: (x - initial_date).days)
data_all['종료일'] = data_all['종료일'].apply(lambda x: (x - initial_date).days)

'''
define variable such as process time, block code..etc. for simulating
'''
data_all['process time'] = list(data_all['종료일'] - data_all['시작일'] + 1)
data_all['호선'] = data_all['호선'].apply(lambda x: str(x))
data_all['블록'] = data_all['블록'].apply(lambda x: str(x))
data_all['block code'] = data_all['호선'] + '_' + data_all['블록']
description_list = []
for i in range(len(data_all)):
    temp = data_all.iloc[i]
    description_list.append(temp['ACT설명'][len(temp['블록']) + 1:])
data_all['data description'] = description_list
print('data pre-processing is done at ', time.time() - start_running)

'''
define shops and number of machines
'''
PE_Shelter = ['1도크쉘터', '2도크쉘터', '3도크쉘터', '의장쉘터', '특수선쉘터', '선행의장1공장쉘터', '선행의장2공장쉘터',
               '선행의장3공장쉘터', '대조립쉘터', '뉴판넬PE장쉘터', '대조립부속1동쉘터', '대조립2공장쉘터', '선행의장6공장쉘터',
               '화공설비쉘터', '판넬조립5부쉘터', '총조립SHOP쉘터', '대조5부쉘터']
server_PE_Shelter = [1, 2, 1, 2, 2, 9, 7, 1, 3, 3, 1, 2, 2, 2, 2, 1, 4, 2]

convert_to_process = {'가공소조립부 1야드': '선각공장', '가공소조립부 2야드': '2야드 중조공장', '대조립1부': '대조립 1공장',
                      '대조립2부': '대조립 2공장', '대조립3부': '2야드 대조립공장', '의장생산부': '해양제관공장',
                      '판넬조립1부': '선각공장', '판넬조립2부': '2야드 판넬공장', '건조1부': ['1도크', '2도크'], '건조2부': '3도크',
                      '건조3부': ['8도크', '9도크'], '선행도장부': ['도장 1공장', '도장 2공장', '도장 3공장', '도장 4공장',
                                                        '도장 5공장', '도장 6공장', '도장 7공장', '도장 8공장',
                                                        '2야드 도장 1공장', '2야드 도장 2공장', '2야드 도장 3공장',
                                                        '2야드 도장 5공장', '2야드 도장 6공장'],
                      '선실생산부': '선실공장', '선행의장부': PE_Shelter, '기장부': PE_Shelter, '의장1부': PE_Shelter,
                      '의장3부': PE_Shelter, '도장1부': '도장1부', '도장2부': '도장2부', '발판지원부': '발판지원부', '외부': '외부',
                      '포항공장부': '포항공장부', '특수선': '특수선', '해양외업생산부': '해양외업생산부'}

shop_list = []
for shop in convert_to_process.values():
    if type(shop) == str:
        if shop not in shop_list:
            shop_list.append(shop)
    else:  # type(process) == list
        for i in range(len(shop)):
            if shop[i] not in shop_list:
                shop_list.append(shop[i])

# machine_dict = {}
# for i in range(len(PE_Shelter)):
#     machine_dict[PE_Shelter[i]] = server_PE_Shelter[i]
#
# for shop in shop_list:
#     if '쉘터' not in shop:
#         if '도크' in shop:
#             machine_dict[shop] = 10
#         elif shop == '외부':
#             machine_dict[shop] = 10000
#         else:
#             machine_dict[shop] = 30
# print('defining converting process and number of machines is done at ', time.time() - start_running)

'''
assemble block data sorting by block code
'''
block_list = list(data_all.drop_duplicates(['block code'])['block code'])

# 각 블록별 activity 개수
activity_num = []
for block_code in block_list:
    temp = data_all[data_all['block code'] == block_code]
    temp_1 = temp.sort_values(by=['시작일'], axis=0, inplace=False)
    temp = temp_1.reset_index(drop=True, inplace=False)
    activity_num.append(len(temp))

# 최대 activity 개수
max_num_of_activity = np.max(activity_num)
print('activity 개수 :', max_num_of_activity)

# SimComponents에 넣어 줄 dataframe(중복된 작업시간 처리)
# activity = assemble을 고려할 activity
columns = pd.MultiIndex.from_product([[i for i in range(max_num_of_activity + 1)],
                                      ['start_time', 'process_time', 'finish_time', 'process', 'description', 'activity']])

data = pd.DataFrame([], columns=columns)
idx = 0  # df에 저장된 block 개수

for block_code in block_list:
    temp = data_all[data_all['block code'] == block_code]
    temp_1 = temp.sort_values(by=['시작일'], axis=0, inplace=False)
    temp = temp_1.reset_index(drop=True)
    data.loc[block_code] = [None for _ in range(len(data.columns))]
    n = 0  # 저장된 공정 개수
    for i in range(0, len(temp)):
        activity = temp['작업부서'][i]
        process = convert_to_process[activity] if type(convert_to_process[activity]) is not list else random.choice(convert_to_process[activity])
        data.loc[block_code][(n, 'start_time')] = temp['시작일'][i]
        data.loc[block_code][(n, 'process_time')] = temp['process time'][i]
        data.loc[block_code][(n, 'finish_time')] = temp['종료일'][i]
        data.loc[block_code][(n, 'process')] = process
        data.loc[block_code][(n, 'description')] = temp['data description'][i]
        data.loc[block_code][(n, 'activity')] = temp['공정공종'][i]
        n += 1

    data.loc[block_code][(n, 'process')] = 'Sink'

print('reassembling data is done at ', time.time() - start_running)
# data.sort_values(by=[(0, 'start_time')], axis=0, inplace=True)

''' ## input data from dataframe to Part class ## '''
parts = OrderedDict()
block_assembly['block code'] = block_assembly['호선'] + '_' + block_assembly['블록']
block_assembly['area'] = block_assembly['길이'] * block_assembly['폭']  # 18개의 block은 area = 0
avg_block_area = np.mean(block_assembly['area'])

for i in range(len(data)):
    part_id = data.index[i]
    if part_id in block_assembly['block code']:
        idx = block_assembly.index[block_assembly['block code'] == part_id].tolist()[0]
        if block_assembly['area'][idx]:
            parts[data.index[i]] = Part(data.index[i], data.iloc[i], area=block_assembly['area'][idx])
        else:
            parts[data.index[i]] = Part(data.index[i], data.iloc[i], area=avg_block_area)
    else:
        parts[data.index[i]] = Part(data.index[i], data.iloc[i], area=avg_block_area)
original_parts = parts.copy()
''' ## BOM data pre-processing'''
block_assembly['호선'] = block_assembly['호선'].apply(lambda x: str(x))
block_assembly['블록'] = block_assembly['블록'].apply(lambda x: str(x))
block_assembly['상위블록'] = block_assembly['상위블록'].apply(lambda x: str(x))
block_assembly['upper block code'] = block_assembly['호선'] + '_' + block_assembly['상위블록']
assembly_list = list(block_assembly.drop_duplicates(['block code'])['block code'])
assembly_upper_list = list(block_assembly.drop_duplicates(['upper block code'])['upper block code'])

'''
adding information about lower block in Part class 
it can contain multiple blocks
'''
for upper_block in assembly_upper_list:
    if upper_block in block_list:
        temp = block_assembly[block_assembly['upper block code'] == upper_block]
        for i in range(len(temp)):
            lower_block = temp.iloc[i]['block code']
            if lower_block in parts:
                parts[upper_block].lower_block_list.append(lower_block)

'''
adding information about upper block in Part class 
'''
for block_code in assembly_list:
    if block_code in block_list:
        temp = block_assembly[block_assembly['block code'] == block_code]
        for i in range(len(temp)):
            upper_block = temp.iloc[i]['upper block code']
            if upper_block in block_list:
                parts[block_code].upper_block = upper_block

upper_block_data = {}

for upper_block in assembly_upper_list:
    if upper_block in parts.keys():
        upper_block_part = parts.pop(upper_block)
        upper_block_data[upper_block] = upper_block_part
len(upper_block_data)
lower_part_list = np.array(list(parts.keys()))
upper_part_list = np.array(list(upper_block_data.keys()))

''' ## modeling ## '''
env = simpy.Environment()
model = {}

monitor = Monitor('../result/event_log_Layout_without_GIS.csv')

source = Source(env, parts, model, monitor, convert_dict=convert_to_process)
for i in range(len(shop_list) + 1):
    if i == len(shop_list):
        model['Sink'] = Sink(env, model, monitor)
    else:
        model[shop_list[i]] = Process(env, shop_list[i], 10000, model, monitor, convert_dict=convert_to_process)

model['Assembly'] = Assembly(env, upper_block_data, source, monitor)

print('modeling is done at ', time.time() - start_running)

start_simulation = time.time()
env.run()
finish_simulation = time.time()

print('#' * 80)
print("Results of simulation")
print('#' * 80)

# 코드 실행 시간
print("data pre-processing : ", start_simulation - start_running)
print("simulation execution time :", finish_simulation - start_simulation)
print("total time : ", finish_simulation - start_running)

print(model['Sink'].parts_rec)
monitor.save_event_tracer()


## VALIDATION
# PLANNED DATA
save_path = '../png/planned'
if not os.path.exists(save_path):
   os.makedirs(save_path)

import math
graph_time = {}
graph_area = {}
for shop in shop_list:
    finish_time = max(math.ceil(model[shop].finish_time), model[shop].finish_time)
    graph_time[shop] = [i for i in range(finish_time)]
    graph_area[shop] = [0 for _ in range(finish_time)]

for i in range(len(data)):
    # 블록 정보
    block_data = data.iloc[i]
    block_name = data.index[i]
    part_area = original_parts[block_name].area

    j = 0  # activity number
    while block_data[(j, 'process')] != 'Sink':
        process = block_data[(j, 'process')]
        graph_area[process][block_data[(j, 'start_time')] : block_data[(j, 'finish_time')] + 2] += part_area

        j += 1

for shop in shop_list:
    index_list = [i for i, value in enumerate(graph_area[shop]) if value > 0]
    min_index = np.min(index_list) if len(index_list) > 0 else 0
    graph_area[shop] = graph_area[shop][min_index:]
    graph_time[shop] = graph_time[shop][min_index:]

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:\Windows\Fonts\H2GTRM.TTF").get_name()
rc('font', family=font_name)

for process in graph_area:
    filepath = '../png/planned/' + process + '_plan.png'
    x = graph_time[process]
    y = graph_area[process]
    plt.plot(x, y)
    ax = plt.axes()
    plt.title(process)
    plt.xlabel('TIME')
    plt.ylabel('AREA')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.savefig(filepath)
    plt.show()

# simulation
save_path = '../png/sim'
if not os.path.exists(save_path):
   os.makedirs(save_path)


for process in model:
    if process == "Assembly" or process == 'Sink':
        continue
    else:
        filepath = '../png/sim/' + process + '_sim.png'
        x = model[process].event_time
        y = model[process].event_area
        plt.plot(x, y)
        ax = plt.axes()
        plt.title(process)
        plt.xlabel('TIME')
        plt.ylabel('AREA')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        plt.savefig(filepath)
        plt.show()
        print(process, model[process].in_process - model[process].parts_sent)

print(0)

