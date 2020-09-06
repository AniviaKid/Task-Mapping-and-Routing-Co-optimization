
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
from libs import init,Get_Neighborhood,Get_mapping_reward,Get_detailed_data,find_start_task,get_sorted_dict
from queue import Queue

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class ActorCritic(nn.Module):#输入：channel*length,4是channel,N是PE 输出：1*action_space和1*1
    def __init__(self, input_channel,input_length, action_space):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(#输入为1*input_channel*input_length
            #nn.Linear(num_inputs, hidden_size),
            #nn.ReLU(),
            #nn.Linear(hidden_size, 1)#1*1
            nn.Conv1d(input_channel,1,input_length,1)#输出为 1*1*1
        )
        
        self.actor = nn.Sequential(
            #nn.Linear(num_inputs, hidden_size),
            #nn.ReLU(),
            #nn.Linear(hidden_size, action_space),#1*action_space
            #nn.Softmax(dim=1),
            #conv1d的输出为1*1*action_space
            nn.Conv1d(input_channel,1,(input_length-action_space+1),1),
            nn.Softmax(dim=2)
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)#.sample()得到的是1*1的二维矩阵tensor，例如[[1]]，再.numpy后可以得到正常的矩阵
        return dist, value


def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns



hyperperiod,num_of_tasks,edges,comp_cost=init('./task graph/N12_autocor.tgff')
adj_matrix,total_needSend,total_needReceive,execution=Get_detailed_data(num_of_tasks,edges,comp_cost)
num_of_rows=4

start_task_list=find_start_task(adj_matrix,num_of_tasks)#入度为0的点的集合
task_graph={}
for start_task in start_task_list:
    q=Queue(maxsize=0)
    q.put(start_task)
    while(q.empty()!=True):
        u=q.get()
        adj_u={}#task_dest - send_size
        for i in range(1,num_of_tasks+1):
            if(adj_matrix[u][i]!=0):
                adj_u[i]=adj_matrix[u][i]
                adj_matrix[u][i]=0
        adj_u=sorted(adj_u.items(), key=lambda x:x[1])#最短任务优先
        for i in adj_u:
            print("visit edge",u,"to",i[0],"size=",i[1])
            q.put(i[0])
            #向task graph中添加task u, task i[0], 以及边u->i[0]
            if(str(u) not in task_graph.keys()):#task u不在task graph中
                task_graph.update({str(u):{'total_needSend':i[1],'out_links':[[str(i[0]),i[1],[],0,0,-1]],'total_needReceive':0,'exe_time':execution[u]}})
            else:#task u在task graph中，仅需要更新出边及相应参数
                task_graph[str(u)]['total_needSend']+=i[1]
                task_graph[str(u)]['out_links'].append([str(i[0]),i[1],[],0,0,-1])
            if(str(i[0]) not in task_graph.keys()):#task i[0]不在task graph中
                task_graph.update({str(i[0]):{'total_needSend':0,'out_links':[],'total_needReceive':i[1],'exe_time':execution[i[0]]}})
            else:#task i[0]在task graph中，仅需要更新接收参数
                task_graph[str(i[0])]['total_needReceive']+=i[1]
            
            #开始为边u->i[0]计算路由
            state=torch.Tensor(np.zeros((1,4,num_of_rows*num_of_rows),dtype=np.int))

print(get_sorted_dict(task_graph))






"""
num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.n

#Hyper params:
hidden_size = 256
lr          = 1e-3
num_steps   = 5

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters())


max_frames   = 20000
frame_idx    = 0
test_rewards = []


state = envs.reset()

while frame_idx < max_frames:

    log_probs = []
    values    = []
    rewards   = []
    masks     = []
    entropy = 0

    # rollout trajectory
    for _ in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        dist, value = model(state)

        action = dist.sample()
        next_state, reward, done, _ = envs.step(action.cpu().numpy())

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        
        state = next_state
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            test_rewards.append(np.mean([test_env() for _ in range(10)]))
            plot(frame_idx, test_rewards)
            
    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    returns = compute_returns(next_value, rewards, masks)
    
    log_probs = torch.cat(log_probs)
    returns   = torch.cat(returns).detach()
    values    = torch.cat(values)

    advantage = returns - values

    actor_loss  = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

test_env(True)
"""