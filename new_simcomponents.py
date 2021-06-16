import simpy
import os
import random
import pandas as pd
import numpy as np
import networkx as nx
from collections import OrderedDict, namedtuple

save_path = './result'
if not os.path.exists(save_path):
   os.makedirs(save_path)

transporter = namedtuple("Transporter", "name, capa, v_loaded, v_unloaded")
workforce = namedtuple("Workforce", "name, skill")
block = namedtuple("Block", "name, location, step")

class Resource(object):
    def __init__(self, env, model, monitor, tp_info=None, wf_info=None, delay_time=None, network=None):
        self.env = env
        self.model = model
        self.monitor = monitor
        self.delay_time = delay_time
        self.network = network

        # resource 할당
        self.tp_store = simpy.Store(env)
        self.wf_store = simpy.FilterStore(env)
        # resource 위치 파악
        self.tp_location = {}
        self.wf_location = {}

        if tp_info is not None:
            for name in tp_info.keys():
                self.tp_location[name] = []
                self.tp_store.put(transporter(name, tp_info[name]["capa"], tp_info[name]["v_loaded"], tp_info[name]["v_unloaded"]))
            # No resource is in resource store -> machine hv to wait
            self.tp_waiting = OrderedDict()
        if wf_info is not None:
            for name in wf_info.keys():
                self.wf_location[name] = []
                self.wf_store.put(workforce(name, wf_info[name]["skill"]))
            # No resource is in resource store -> machine hv to wait
            self.wf_waiting = OrderedDict()


    def request_tp(self, current_process):
        tp, waiting = False, False

        if len(self.tp_store.items) > 0:  # 만약 tp_store에 남아있는 transporter가 있는 경우
            tp = yield self.tp_store.get()
            self.monitor.record(self.env.now, None, None, part_id=None, event="tp_going_to_requesting_process",
                                resource=tp.name)
            return tp, waiting
        else:  # transporter가 전부 다른 공정에 가 있을 경우
            tp_location_list = []
            for name in self.tp_location.keys():
                if not self.tp_location[name]:
                    continue
                tp_current_location = self.tp_location[name][-1]
                if len(self.model[tp_current_location].tp_store.items) > 0:  # 해당 공정에 놀고 있는 tp가 있다면
                    tp_location_list.append(self.tp_location[name][-1])

            if len(tp_location_list) == 0:  # 유휴 tp가 없는 경우
                waiting = True
                return tp, waiting

            else:  # 유휴 tp가 있어 part을 실을 수 있는 경우
                # tp를 호출한 공정 ~ 유휴 tp 사이의 거리 중 가장 짧은 데에서 호출
                distance = []
                for location in tp_location_list:
                    called_distance = self.network[location][current_process]
                    distance.append(called_distance)
                # distance = list(map(lambda i: self.network.get_shortest_path_distance(tp_location_list[i], current_process), tp_location_list))
                location_idx = distance.index(min(distance))
                location = tp_location_list[location_idx]
                tp = yield self.model[location].tp_store.get()
                self.monitor.record(self.env.now, None, None, part_id=None, event="tp_going_to_requesting_process", resource=tp.name)
                yield self.env.timeout(distance[location_idx]/tp.v_unloaded)  # 현재 위치까지 오는 시간
                return tp, waiting


class Part:
    def __init__(self, name, env, process_data, processes, monitor, resource=None, from_to_matrix=None, length=0,
                 breadth=0, child=None, parent=None, stocks=None):
        self.name = name
        self.env = env
        self.data = process_data
        self.processes = processes
        self.monitor = monitor
        self.resource = resource
        self.from_to_matrix = from_to_matrix
        self.area = length * breadth
        self.size = min(length, breadth)
        self.child = child
        self.parent = parent
        self.stocks = stocks

        self.in_child = []
        self.location = []
        self.part_store = simpy.Store(env)
        self.num_clone = 0
        self.process_delay = env.event()
        self.resource_delay = env.event()

    def _sourcing(self):
        if self.child is not None and len(self.in_child):
            step = 0
            while True:
                if step == 0:  # 가장 처음이면 part creating
                    part = block(self.name, "source", step)
                    self.part_store.put(part)
                    self.monitor.record(self.env.now, None, None, part_id=self.name, event="part_created")
                else:
                    if len(self.part_store.items):  # store에 유휴 part가 있으면
                        part = yield self.part_store.get()
                    else:  # 겹치는 일정으로 인해 유휴 part를 만들어야 할 때  -> part.name_1
                        self.num_clone += 1
                        part = block(self.name + "_{0}".format(self.num_clone), self.location[-1], step)

                process = self.processes[self.data[(step, 'process')]]
                if process.name == 'Sink':
                    self.processes['Sink'].put(part)
                    part.location = 'Sink'
                    break
                else:
                    yield self.env.timeout(self.data[(step,'start_time')] - self.env.now)

                    work = self.data[(step, 'work')]  # 공정공종

                    # 처음이거나 tp 사용 단계가 아닌 경우 tp 사용 않고 바로 다음 공정으로
                    if step == 0 or work in ['C', 'C_0', 'C_1']:
                        self.processes[process].buffer_to_machine.put(part)
                        part.location = process
                    else:  ## tp를 사용해야 하는 경우
                        self._tp(process, part.location, part)

    def return_to_part(self, part):
        next_start_time = part.data[(part.step + 1, 'start_time')]
        if next_start_time - self.env.now > 1:  # 적치장 가야 함
            next_process = part.data[(part.step + 1, 'process')]
            stock = self._find_stock(next_process)
            if part.data[(part.step, 'work')] in ['C', 'C_0', 'C_1']:  # tp 불필요
                self.stocks[stock].put(part, next_start_time)
                self.monitor.record(self.env.now, part.location, None, part_id=part.name, event="go_to_stock")
            else:
                self._tp(next_process, part.location, part)
        else:
            self.part_store.put(part)

    def _tp(self, next_process, current_process, part):
        self.monitor.record(self.env.now, current_process, None, part_id=part.name, event="tp_request")
        tp, waiting = False, True
        while waiting:
            tp, waiting = yield self.env.process(self.resource.request_tp(current_process))
            if not waiting:
                break
            # if waiting is True == All tp is moving == process hv to delay
            else:
                self.resource.tp_waiting[current_process] = self.env.event()
                self.monitor.record(self.env.now, current_process, None, part_id=part.name,
                                    event="delay_start_cus_no_tp")
                yield self.resource.tp_waiting[self.name]
                self.monitor.record(self.env.now, current_process, None, part_id=part.name,
                                    event="delay_finish_cus_yes_tp")
                continue
        if tp:
            self.monitor.record(self.env.now, current_process, None, part_id=part.name, event="tp_going_to_next_process",
                                resource=tp.name)
            distance_to_move = self.from_to_matrix[current_process][next_process]
            yield self.env.timeout(distance_to_move / tp.v_loaded)
            self.monitor.record(self.env.now, next_process, None, part_id=part.name,
                                event="tp_finished_transferred_to_next_process", resource=tp.name)
            next_process.buffer_to_machine.put(part.name)
            self.monitor.record(self.env.now, current_process, None, part_id=part.name,
                                event="part_transferred_to_next_process_with_tp")
            next_process.tp_store.put(tp)
            part.location = next_process.name
            self.resource.tp_location[tp.name].append(next_process)
            # 가용한 tp 하나 발생 -> delay 끝내줌
            if len(self.resource.tp_waiting) > 0:
                self.resource.tp_waiting.popitem(last=False)[1].succeed()

    def _find_stock(self, next_process):
        stocks = list(self.stocks.keys())
        next_stock = None
        shortest_path = 0
        for idx in range(len(stocks)):
            distance = self.from_to_matrix[next_process][stocks[idx]]
            if idx:
                if shortest_path > distance:  # 이번 적치장까지의 거리가 더 가까운 경우
                    next_stock = stocks[idx]
                    shortest_path = distance
                elif shortest_path == distance:  # 두 적치장까지의 거리가 같은 경우 -> 지금 쌓여 있는 블록 수가 더 적은 데로
                    current = len(self.stocks[next_stock].store)
                    alter = len(self.stocks[stocks[idx]].store)
                    next_stock = stocks[idx] if current > alter else next_stock
                else:  # 기존 적치장까지의 거리가 더 가까운 경우 현행 유지
                    continue

        return next_stock


class Sink:  ## 후우우... 
    def __init__(self, env, processes, parts, monitor):
        self.env = env
        self.name = 'Sink'
        self.processes = processes
        self.parts = parts
        self.monitor = monitor

        # self.tp_store = simpy.FilterStore(env)  # transporter가 입고 - 출고 될 store
        self.parts_rec = 0
        self.last_arrival = 0.0
        self.completed_part = []

    def put(self, part):
        # if part.upper_block is None:
        self.parts_rec += 1
        self.last_arrival = self.env.now
        self.monitor.record(self.env.now, self.name, None, part_id=part.id, event="completed")
        self.completed_part.append(part.id)

class StockYard:
    def __init__(self, env, name, parts, monitor):
        self.name = name
        self.env = env
        self.parts = parts
        self.monitor = monitor

        self.stock_yard = OrderedDict()

    def put(self, part, out_time):
        self.monitor.record(self.env.now, self.name, None, part_id=part, event="Stock_in")
        part.location = self.name
        self.stock_yard[part.name] = [out_time, part]

    def out_to_part(self):
        while True:
            self.stock_yard = OrderedDict(sorted(self.stock_yard.items(), key=lambda x: x[1][0]))
            part, next_start = self.stock_yard.popitem(last=False)[1][1], self.stock_yard.popitem(last=False)[1][0]
            yield self.env.timeout(next_start - self.env.timeout)

            self.parts[part.name].return_to_part(part)
            self.monitor.record(self.env.now, self.name, None, part_id=part.name, event="Stock_out")


class Monitor(object):
    def __init__(self, filepath):
        self.filepath = filepath  ## Event tracer 저장 경로

        self.time=[]
        self.event=[]
        self.part_id=[]
        self.process=[]
        self.subprocess=[]
        self.resource = []

    def record(self, time, process, subprocess, part_id=None, event=None, resource=None):
        self.time.append(time)
        self.event.append(event)
        self.part_id.append(part_id)
        self.process.append(process)
        self.subprocess.append(subprocess)
        self.resource.append(resource)

    def save_event_tracer(self):
        event_tracer = pd.DataFrame(columns=['Time', 'Event', 'Part', 'Process', 'SubProcess'])
        event_tracer['Time'] = self.time
        event_tracer['Event'] = self.event
        event_tracer['Part'] = self.part_id
        event_tracer['Process'] = self.process
        event_tracer['SubProcess'] = self.subprocess
        event_tracer['Resource'] = self.resource
        event_tracer.to_csv(self.filepath, encoding='utf-8-sig')

        return event_tracer




