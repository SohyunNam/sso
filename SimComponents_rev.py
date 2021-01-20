import simpy
import os
import pandas as pd
import numpy as np

save_path = './result'
if not os.path.exists(save_path):
    os.makedirs(save_path)


class Part(object):
    def __init__(self, name, data):
        # 해당 Part의 이름
        self.id = name
        # 작업 시간 정보
        self.data = data
        # 작업을 완료한 공정의 수
        self.step = 0


class Source(object):
    def __init__(self, env, parts, model, monitor):
        self.env = env
        self.name = 'Source'
        self.parts = parts  ## Part 클래스로 모델링 된 Part들이 list 형태로 저장
        self.model = model
        self.monitor = monitor

        self.action = env.process(self.run())

    def run(self):
        while True:
            part = self.parts.pop(0)

            IAT = part.data[(0, 'start_time')] - self.env.now
            if IAT > 0:
                yield self.env.timeout(part.data[(0, 'start_time')] - self.env.now)

            # record: part_created
            self.monitor.record(self.env.now, self.name, None, part_id=part.id, event="part_created")

            # next process
            next_process = part.data[(part.step, 'process')]

            next_server, next_queue = self.model[next_process].get_num_of_part()
            if next_server + next_queue >= self.model[next_process].qlimit:
                self.model[next_process].waiting[part.id] = self.env.event()
                self.model[next_process].delay_part_id.append(part.id)
                self.monitor.record(self.env.now, self.name, None, part_id=part.id, event="delay_start")

                yield self.model[next_process].waiting[part.id]
                self.monitor.record(self.env.now, self.name, None, part_id=part.id, event="delay_finish")

            # record: part_transferred
            self.monitor.record(self.env.now, self.name, None, part_id=part.id, event="part_transferred")
            ## self.model[next_process].put(part)

            if len(self.parts) == 0:
                print("all parts are sent at : ", self.env.now)
                break


class Process(object):
    def __init__(self, env, name, machine_num, model, monitor, process_time=None, capacity=float('inf'),
                 routing_logic='cyclic', priority=None, capa_to_machine=float('inf'), capa_to_process=float('inf')):
        self.env = env
        self.name = name
        self.model = model
        self.monitor = monitor
        self.process_time = process_time[self.name] if process_time is not None else [None for _ in range(machine_num)]
        self.capa = capacity
        self.machine_num = machine_num
        self.routing_logic = routing_logic
        self.priority = priority[self.name] if priority is not None else [1 for _ in range(machine_num)]

        self.buffer_to_machine = simpy.Store(env, capacity=capa_to_machine)
        self.buffer_to_process = simpy.Store(env, capacity=capa_to_process)
        self.machine = [Machine(env, '{0}_{1}'.format(self.name, i + 1), process_time=self.process_time[i],
                                priority=self.priority[i], out=self.buffer_to_process, monitor=monitor) for i in
                        range(self.machine_num)]

        self.parts_sent = 0
        self.server_idx = 0
        self.waiting = {}
        self.delay_part_id = []
        self.len_of_server = []

        env.process(self._to_machine())
        env.process(self._to_process())

    def _to_machine(self):
        while True:
            part = yield self.buffer_to_machine.get()
            self.monitor.record(self.env.now, self.name, None, part_id=part.id, event="Process_entered")
            ## Rouring logic 추가 할 예정
            self.server_idx = 0 if (self.parts_sent == 0) or (
                    self.server_idx == self.machine_num - 1) else self.server_idx + 1



    def _to_process(self):
        print(0)


class Machine(object):
    def __init__(self, env, name, process_time, priority, out, monitor):
        self.env = env
        self.name = name
        self.process_time = process_time
        self.priority = priority
        self.out = out
        self.monitor = monitor

        env.process(self.work())
        self.queue = simpy.Store(env, capacity=1)
        self.working_start = 0.0
        self.total_time = 0.0
        self.total_working_time = 0.0
        self.working = False

    def work(self):
        while True:
            part = yield self.queue.get()
            self.working = True

            # process_time
            if self.process_time == None:  # part에 process_time이 미리 주어지는 경우
                proc_time = part.data[(part.step, "process_time")]
            else:  # service time이 정해진 경우 --> 1) fixed time / 2) Stochastic-time
                proc_time = self.process_time if type(self.process_time) == float else self.process_time()

            # working
            self.working_start = self.env.now
            yield self.env.timeout(proc_time)

            # delay: buffer_to_process의 capacity가 차 있을 때
            if len(self.out) == self.out.capacity:
                print(0)



class Sink(object):
    def __init__(self, env, monitor):
        self.env = env
        self.name = 'Sink'
        self.monitor = monitor

        # self.tp_store = simpy.FilterStore(env)  # transporter가 입고 - 출고 될 store
        self.parts_rec = 0
        self.last_arrival = 0.0

    def put(self, part):
        self.parts_rec += 1
        self.last_arrival = self.env.now
        self.monitor.record(self.env.now, self.name, None, part_id=part.id, event="completed")


class Monitor(object):
    def __init__(self, filepath):
        self.filepath = filepath  ## Event tracer 저장 경로

        self.time=[]
        self.event=[]
        self.part_id=[]
        self.process=[]
        self.subprocess=[]

    def record(self, time, process, subprocess, part_id=None, event=None):
        self.time.append(time)
        self.event.append(event)
        self.part_id.append(part_id)
        self.process.append(process)
        self.subprocess.append(subprocess)

    def save_event_tracer(self):
        event_tracer = pd.DataFrame(columns=['Time', 'Event', 'Part', 'Process', 'SubProcess'])
        event_tracer['Time'] = self.time
        event_tracer['Event'] = self.event
        event_tracer['Part'] = self.part_id
        event_tracer['Process'] = self.process
        event_tracer['SubProcess'] = self.subprocess
        event_tracer.to_csv(self.filepath)

        return event_tracer

