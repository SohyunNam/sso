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
    def __init__(self, name, env, process_data, processes, monitor, resource=None, from_to_matrix=None, size=0,
                 area=0, child=None, parent=None, stocks=None):
        self.name = name
        self.env = env
        self.data = process_data
        self.processes = processes
        self.monitor = monitor
        self.resource = resource
        self.from_to_matrix = from_to_matrix
        self.area = area
        self.size = size
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


class Sink:
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
        if self.parts[part.name].parent:  # 상위 블록이 있는 경우
            parent_block = self.parts[part.name].parent
            self.parts[parent_block].in_child.append(part.name)

        self.parts_rec += 1
        self.last_arrival = self.env.now
        self.monitor.record(self.env.now, self.name, None, part_id=part.name, event="completed")


class Process:
    def __init__(self, env, name, machine_num, processes, parts, monitor, resource=None,
                 process_time=None, capacity=float('inf'), routing_logic='cyclic', priority=None,
                 capa_to_machine=float('inf'), capa_to_process=float('inf'), MTTR=None, MTTF=None,
                 initial_broken_delay=None, delay_time=None, workforce=None, convert_dict=None):

        # input data
        self.env = env
        self.name = name
        self.processes = processes
        self.parts = parts
        self.monitor = monitor
        self.resource = resource
        self.capa = capacity
        self.machine_num = machine_num
        self.routing_logic = routing_logic
        self.process_time = process_time[self.name] if process_time is not None else [None for _ in
                                                                                      range(machine_num)]
        self.priority = priority[self.name] if priority is not None else [1 for _ in range(machine_num)]
        self.MTTR = MTTR[self.name] if MTTR is not None else [None for _ in range(machine_num)]
        self.MTTF = MTTF[self.name] if MTTF is not None else [None for _ in range(machine_num)]
        self.initial_broken_delay = initial_broken_delay[self.name] if initial_broken_delay is not None else [None
                                                                                                              for _
                                                                                                              in
                                                                                                              range(
                                                                                                                  machine_num)]
        self.delay_time = delay_time[name] if delay_time is not None else None
        self.workforce = workforce[self.name] if workforce is not None else [False for _ in
                                                                             range(machine_num)]  # workforce 사용 여부
        self.converting = convert_dict

        # variable defined in class
        self.in_process = 0
        self.parts_sent = 0
        self.parts_sent_to_machine = 0
        self.machine_idx = 0
        self.len_of_server = []
        self.waiting_machine = OrderedDict()
        self.waiting_pre_process = OrderedDict()
        self.area_used = 0.0
        self.finish_time = 0.0
        self.start_time = 0.0
        self.event_area = []
        self.event_time = []

        # buffer and machine
        self.buffer_to_machine = simpy.Store(env, capacity=capa_to_machine)
        self.buffer_to_process = simpy.Store(env, capacity=capa_to_process)
        self.machine = [Machine(env, '{0}_{1}'.format(self.name, i), self.name, self.parts, self.resource,
                                process_time=self.process_time[i], priority=self.priority[i],
                                waiting=self.waiting_machine, monitor=monitor, MTTF=self.MTTF[i], MTTR=self.MTTR[i],
                                initial_broken_delay=self.initial_broken_delay[i],
                                workforce=self.workforce[i]) for i in range(self.machine_num)]
        # resource
        self.tp_store = simpy.Store(self.env)
        self.wf_store = simpy.Store(self.env)

        # get run functions in class
        env.process(self.to_machine())

    def to_machine(self):
        while True:
            routing = Routing(self.machine, priority=self.priority)
            if self.delay_time is not None:
                delaying_time = self.delay_time if type(self.delay_time) == float else self.delay_time()
                yield self.env.timeout(delaying_time)
            part = yield self.buffer_to_machine.get()
            if self.in_process == 0:
                self.start_time = self.env.now
            self.in_process += 1
            self.area_used += part.area
            self.event_area.append(self.area_used)
            self.event_time.append(self.env.now)
            self.monitor.record(self.env.now, self.name, None, part_id=part.id, event="Process_entered")

            ## Rouring logic 추가 할 예정
            if self.routing_logic == 'priority':
                self.machine_idx = routing.priority()
            else:
                self.machine_idx = 0 if (self.parts_sent_to_machine == 0) or (
                            self.machine_idx == self.machine_num - 1) else self.machine_idx + 1

            self.monitor.record(self.env.now, self.name, None, part_id=part.id, event="routing_ended")
            self.machine[self.machine_idx].machine.put(part)
            self.parts_sent_to_machine += 1

            # finish delaying of pre-process
            if (len(self.buffer_to_machine.items) < self.buffer_to_machine.capacity) and (
                    len(self.waiting_pre_process) > 0):
                self.waiting_pre_process.popitem(last=False)[1].succeed()  # delay = (part_id, event)


class Machine:
    def __init__(self, env, name, process_name, parts, resource, process_time, priority, waiting, monitor, MTTF,
                 MTTR,
                 initial_broken_delay, workforce):
        # input data
        self.env = env
        self.name = name
        self.process_name = process_name
        self.parts = parts
        self.resource = resource
        self.process_time = process_time
        self.priority = priority
        self.waiting = waiting
        self.monitor = monitor
        self.MTTR = MTTR
        self.MTTF = MTTF
        self.initial_broken_delay = initial_broken_delay
        self.workforce = workforce

        # variable defined in class
        self.machine = simpy.Store(env)
        self.working_start = 0.0
        self.total_time = 0.0
        self.total_working_time = 0.0
        self.working = False  # whether machine's worked(True) or idled(False)
        self.broken = False  # whether machine is broken or not
        self.unbroken_start = 0.0
        self.planned_proc_time = 0.0

        # broke and re-running
        self.residual_time = 0.0
        self.broken_start = 0.0
        if self.MTTF is not None:
            mttf_time = self.MTTF if type(self.MTTF) == float else self.MTTF()
            self.broken_start = self.unbroken_start + mttf_time
        # get run functions in class
        self.action = env.process(self.work())
        # if (self.MTTF is not None) and (self.MTTR is not None):
        #     env.process(self.break_machine())

    def work(self):
        while True:
            self.broken = True
            part = yield self.machine.get()
            # if part.id == "A0001_H11P3":
            #     print(0)
            self.working = True
            wf = None
            # process_time
            if self.process_time == None:  # part에 process_time이 미리 주어지는 경우
                proc_time = part.data[(part.step, "process_time")]
            else:  # service time이 정해진 경우 --> 1) fixed time / 2) Stochastic-time
                proc_time = self.process_time if type(self.process_time) == float else self.process_time()
            self.planned_proc_time = proc_time

            if self.workforce is True:
                resource_item = list(map(lambda item: item.name, self.resource.wf_store.items))
                self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                    event="workforce_request", resource=resource_item)
                while len(self.resource.wf_store.items) == 0:
                    self.resource.wf_waiting[part.id] = self.env.event()
                    self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                        event="delay_start_machine_cus_no_resource")
                    yield self.resource.wf_waiting[part.id]  # start delaying

                self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                    event="delay_finish_machine_cus_yes_resource")
                wf = yield self.resource.wf_store.get()
                self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                    event="workforce get in the machine", resource=wf.name)
            while proc_time:
                if self.MTTF is not None:
                    self.env.process(self.break_machine())
                try:
                    self.broken = False
                    ## working start
                    self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                        event="work_start")
                    self.working_start = self.env.now
                    yield self.env.timeout(proc_time)

                    ## working finish
                    self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                        event="work_finish")
                    self.total_working_time += self.env.now - self.working_start
                    self.broken = True
                    proc_time = 0.0

                except simpy.Interrupt:
                    self.broken = True
                    self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                        event="machine_broken")
                    print('{0} is broken at '.format(self.name), self.env.now)
                    proc_time -= self.env.now - self.working_start
                    if self.MTTR is not None:
                        repair_time = self.MTTR if type(self.MTTR) == float else self.MTTR()
                        yield self.env.timeout(repair_time)
                        self.unbroken_start = self.env.now
                    self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                        event="machine_rerunning")
                    print(self.name, 'is solved at ', self.env.now)
                    self.broken = False

                    mttf_time = self.MTTF if type(self.MTTF) == float else self.MTTF()
                    self.broken_start = self.unbroken_start + mttf_time

            self.working = False

            if self.workforce is True:
                self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                    event="workforce_used_out", resource=wf.name)
                self.resource.wf_store.put(wf)
                self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                    event="workforce get out the machine", resource=wf.name)
                if (len(self.resource.wf_store.items) > 0) and (len(self.resource.wf_waiting) > 0):
                    self.resource.wf_waiting.popitem(last=False)[1].succeed()  # delay = (part_id, event)

            # # start delaying at machine cause buffer_to_process is full
            # if len(self.out.items) == self.out.capacity:
            #     self.waiting[part.id] = self.env.event()
            #     self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
            #                         event="delay_start_machine")
            #     yield self.waiting[part.id]  # start delaying
            #     self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
            #                         event="delay_finish_machine")

            # transfer to 'to_process' function
            self.parts[part.name].return_to_part(part)
            self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                event="part_transferred_to_out_buffer")

            self.total_time += self.env.now - self.working_start

    def break_machine(self):
        if (self.working_start == 0.0) and (self.initial_broken_delay is not None):
            initial_delay = self.initial_broken_delay if type(
                self.initial_broken_delay) == float else self.initial_broken_delay()
            yield self.env.timeout(initial_delay)
        residual_time = self.broken_start - self.working_start
        if (residual_time > 0) and (residual_time < self.planned_proc_time):
            yield self.env.timeout(residual_time)
            self.action.interrupt()
        else:
            return


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


class Monitor:
    def __init__(self, filepath):
        self.filepath = filepath  ## Event tracer 저장 경로

        self.time = []
        self.event = []
        self.part_id = []
        self.process = []
        self.subprocess = []
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


class Routing:
    def __init__(self, server_list=None, priority=None):
        self.server_list = server_list
        self.idx_priority = np.array(priority)

    def priority(self):
        i = min(self.idx_priority)
        idx = 0
        while i <= max(self.idx_priority):
            min_idx = np.argwhere(self.idx_priority == i)  # priority가 작은 숫자의 index부터 추출
            idx_min_list = min_idx.flatten().tolist()
            # 해당 index list에서 machine이 idling인 index만 추출
            idx_list = list(filter(lambda j: (self.server_list[j].working == False), idx_min_list))
            if len(idx_list) > 0:  # 만약 priority가 높은 machine 중 idle 상태에 있는 machine이 존재한다면
                idx = random.choice(idx_list)
                break
            else:  # 만약 idle 상태에 있는 machine이 존재하지 않는다면
                if i == max(self.idx_priority):  # 그 중 모든 priority에 대해 machine이 가동중이라면
                    idx = random.choice([j for j in range(len(self.idx_priority))])  # 그냥 무작위 배정
                    # idx = None
                    break
                else:
                    i += 1  # 다음 priority에 대하여 따져봄
        return idx

    def first_possible(self):
        idx_possible = random.choice(len(self.server_list))  # random index로 초기화 - 모든 서버가 가동중일 때, 서버에 random하게 파트 할당
        for i in range(len(self.server_list)):
            if self.server_list[i].working is False:  # 만약 미가동중인 server가 존재할 경우, 해당 서버에 part 할당
                idx_possible = i
                break
        return idx_possible

