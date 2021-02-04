import pandas as pd
import scipy.stats as st
import numpy as np
import simpy
import time
import functools
from SimComponents_rev import Part, Sink, Process, Source, Monitor

workforces = 1
SIM_TIME = 50000
process_list = ['Process1', 'Process2', 'Process3']
for i in range(workforces):
    env = simpy.Environment()
