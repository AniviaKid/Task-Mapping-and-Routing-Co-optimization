
import math
import random

#import gym  
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
import time
from libs import init,Get_full_route_by_XY,Environment,check_if_Done,Critic
from queue import Queue





"""
task_graph={'1': {'total_needSend': 48, 'out_links': [['2', 48, [], 0, 0, -1]], 'total_needReceive': 0, 'exe_time': 10}, '2': {'total_needSend': 0, 'out_links': [], 'total_needReceive': 48, 'exe_time': 10}}
MapResult=[-1,4,13,10,11]

state_tensor=torch.Tensor(np.zeros((1,4,4*4),dtype=np.int))
cur_position=4
partRoute=[]
state=[state_tensor,cur_position,partRoute]
end_state,tmp_reward,tmp_done=check_if_Done(state,source_position=4,dest_position=MapResult[2],num_of_rows=4,task_graph=task_graph,fullRouteFromRL={},task_source=1,task_dest=2,MapResult=MapResult)
if(tmp_done==False):
    print("exe action")
    next_state,reward,done=Environment(state,action=1,source_position=4,dest_position=MapResult[2],num_of_rows=4,task_graph=task_graph,fullRouteFromRL={},task_source=1,task_dest=2,MapResult=MapResult)
    print(state[0])
    print(state[1])
    print(state[2])
    print(next_state[0])
    print(next_state[1])
    print(next_state[2])
    state=next_state
    print(state[0])
    print(state[1])
    print(state[2])
    print(reward)
    print(done)




"""
#state=torch.Tensor(np.random.rand(1,1,2))
state=torch.Tensor(np.random.rand(1,4,16))
#print(state)
softmax=nn.Softmax(dim=2)
critic=Critic(4,16)
value=critic(state)
print(value,type(value),value.shape)
"""
probs=softmax(state)
#print(probs,type(probs),probs.shape)
#print(probs[0][0],type(probs[0][0]))
dist=Categorical(probs[0][0])
print(dist)
action=dist.sample()
print(action,type(action),action.shape)
tmp=int(action)
print("test tmp",tmp,type(tmp))
log_prob=dist.log_prob(action)
print(log_prob,log_prob.shape)
"""

#hyperperiod,num_of_tasks,edges,comp_cost=init('./task graph/N4_test.tgff')



