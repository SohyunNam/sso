import simpy
import pandas as pd
import numpy as np
import random
from SimComponents_for_Team2 import Resource, Part, Sink, StockYard, Monitor

raw_data = pd.read_csv('../data/Team2_data.csv')
raw_data = raw_data.fillna("finish")
columns = pd.MultiIndex.from_product([[i for i in range(9)], ['start_time', 'process_time', 'process', 'area']])
data = pd.DataFrame(columns=columns, index=raw_data['Unnamed: 0_level_0'][1:])

# data pre-processing
for i in range(1, len(raw_data)):
    temp = list(raw_data.iloc[i][1:])
    process_data = [None for _ in range(36)]
    idx = 0
    while len(temp):
        start_time = temp.pop(0)
        duration = temp.pop(0)
        process = temp.pop(0)

        if process == "finish":
            if 'Sink' not in process_data:
                start_time = None
                duration = None
                process = 'Sink'
                area = 0
            else:
                continue
        elif process == "Fabrication":
            area = random.uniform(37.5*0.8, 37.5*1.2)
        elif process == "Unit_Assmbly":
            area = random.uniform(150*0.8, 150*1.2)
        elif process == "Sub_Assembly" or process == "Grand_Assembly" or process == "Block_Outfitting" or \
                process == "Painting":
            area = random.uniform(300*0.8, 300*1.2)
        elif process == "Sink":
            start_time = None
            duration = None
            area = 0
        else:
            area = random.uniform(360*0.8, 360*1.2)

        process_data[idx*4] = start_time
        process_data[idx*4+1] = duration
        process_data[idx*4+2] = process
        process_data[idx*4+3] = area
        idx += 1
    data.iloc[i-1] = process_data

process_list = ["Fabrication", "Unit_Assembly", "Sub_Assembly", "Grand_Assembly", "Block_Outfitting", "Painting",
                "PE_Outfitting", "PE", "PE_Painting", "Shop_PE", "Shot_PE_Outfitting", "Shop_PE_Painting"]

monitor = Monitor('../result/Team2_result.xlsx')
parts = {}
processes = {}

stock_yard = {}


print(0)