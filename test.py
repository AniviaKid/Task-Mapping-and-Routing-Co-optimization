
import math
import random

#import gym  
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
import time
#from libs import init,Get_full_route_by_XY,Environment,check_if_Done,Critic,Get_detailed_data,Get_rand_computation_ability2,Get_mapping_exe_time,computeContention,Get_link_index_by_route,Update_link_set
from queue import Queue

import datetime
from routing import improved_routeCompute
#from MyOnlineCompute import onlineTimeline

"""
task_graph={"1": {"total_needSend": 3, "out_links": [["2", 2, [], 0, 0, -1], ["3", 1, [[0, "S"]], 1, 2, 1]], "total_needReceive": 0, "exe_time": 1}, "2": {"total_needSend": 7, "out_links": [["4", 3, [[0, "E"]], 4, 7, 2], ["5", 4, [], 0, 0, -1]], "total_needReceive": 2, "exe_time": 3}, "3": {"total_needSend": 2, "out_links": [["6", 2, [], 0, 0, -1]], "total_needReceive": 1, "exe_time": 2}, "4": {"total_needSend": 2, "out_links": [["7", 2, [], 0, 0, -1]], "total_needReceive": 3, "exe_time": 5}, "5": {"total_needSend": 2, "out_links": [["7", 2, [[0, "E"]], 6, 8, 3]], "total_needReceive": 4, "exe_time": 2}, "6": {"total_needSend": 1, "out_links": [["7", 1, [[2, "N"], [0, "E"]], 6, 7, 4]], "total_needReceive": 2, "exe_time": 4}, "7": {"total_needSend": 0, "out_links": [], "total_needReceive": 5, "exe_time": 1}}
"""

class link_item():#可以根据在list中的下标索引到它连接的是哪两个PE
    def __init__(self):
        #记录这个link的timeline，list中的每个元素是list，形式为[task_source,task_dest,start_time,end_time]
        self.timeline=[]

a=10
b=4
c=a/b
print(c,type(c))
"""
link_set=[]
total_link_num=(num_of_rows-1+num_of_rows)*(num_of_rows-1)+num_of_rows-1
for i in range(0,total_link_num):
    tmp=link_item()
    link_set.append(tmp)

edge_1_2_route=[[5,'E'],[6,'S'],[10,'E']]#[0,20]
edge_1_3_route=[[5,'E'],[6,'N'],[2,'E']]#[0,30]
edge_3_4_route=[[3,'W'],[2,'S'],[6,'S'],[10,'S'],[14,'E']]#[end_time_3,end_time_3+20]

for i in edge_1_2_route:
    tmp_link=Get_link_index_by_route(i,num_of_rows)
    link_set[tmp_link].timeline.append([0,20])

T_1_3=computeContention(edge_1_3_route,link_set,num_of_rows,0,30)
Update_link_set(edge_1_3_route,link_set,num_of_rows,T_1_3,30+T_1_3)
start_time_of_3=30+T_1_3
print(computeContention(edge_3_4_route,link_set,num_of_rows,start_time_of_3,start_time_of_3+20))
"""

