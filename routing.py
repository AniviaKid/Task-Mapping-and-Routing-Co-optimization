
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
from libs import init,Get_Neighborhood,Get_mapping_reward,Get_detailed_data,find_start_task,get_sorted_dict,ActorCritic,Get_full_route_by_XY
from queue import Queue
from OnlineCompute1 import onlineTimeline







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





def routeCompute(adj_matrix,num_of_tasks,execution,num_of_rows,MapResult):
    start_task_list=find_start_task(adj_matrix,num_of_tasks)#入度为0的点的集合
    task_graph={}
    fullRouteFromRL={}#key-value表示从task_key到task_value的route全部是由RL计算的
    partRouteFromRL=[]#它的元素就是json里的route格式，如[0, "S"]，当前位置+下一步移动方向

    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    for start_task in start_task_list:
        q=Queue(maxsize=0)
        q.put(start_task)
        while(q.empty()!=True):#BFS
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
                #state的四个channel,从0-3以此为N,S,W,E
                #state_tensor=torch.Tensor(np.zeros((1,4,num_of_rows*num_of_rows),dtype=np.int))
                #cur_position_in_calc=mapResult[u]
                #传进RL之前先check一次，确保当前的state不是end state，至少能执行一次action
                #传进网络中计算action....
                #done之后要根据partRoute更新fullRoute和taskgraph

    print(get_sorted_dict(task_graph))

#state为[state_tensor,cur_position,partRouteFromRL]，传进来的partRoute的格式是直接的路由表，没有第一位第二位的task
def check_if_Done(state,source_position,dest_position,num_of_rows,task_graph,fullRouteFromRL,task_source,task_dest,MapResult):#检查当前的state是否已经结束，结束了的话直接把end_state,reward,done=True返回
    next_state_tensor=state[0]
    next_position=-1
    next_partRoute=state[2]

    cur_row=int(state[1]/num_of_rows)
    cur_col=state[1]%num_of_rows
    dest_row=int(dest_position/num_of_rows)
    dest_col=dest_position%num_of_rows

    flag=False

    if(cur_row==dest_row or cur_col==dest_col):#结束，先更新tensor
        flag=True
        if(cur_row==dest_row and cur_col==dest_col):#考虑到一开始两个task就被map到同一个PE的情况
            next_state_tensor=state[0]
        elif(cur_row==dest_row):
            if(cur_col<dest_col):#向East走
                for i in range(cur_col,dest_col):#更新tensor
                    next_state_tensor[0][3][cur_row*num_of_rows+i]=1
            else:#向West走
                for i in range(cur_col,dest_col,-1):#更新tensor
                    next_state_tensor[0][2][cur_row*num_of_rows+i]=1
        elif(cur_col==dest_col):
            if(cur_row<dest_row):#向South走
                for i in range(cur_row,dest_row):#更新tensor
                    next_state_tensor[0][1][i*num_of_rows+cur_col]=1
            else:#向North走
                for i in range(cur_row,dest_row,-1):
                    next_state_tensor[0][0][i*num_of_rows+cur_col]=1
    
    if(flag==False):#没有结束
        return [],0,False
    else:#结束
        #更新position
        next_position=dest_position
        #更新partRoute，此时的partRoute就是这一条链路全部的路由表，可以直接传进onlineCompute
        next_partRoute=Get_full_route_by_XY(state[2],source_position,dest_position,num_of_rows)
        #处理参数，传进onlineCompute计算pending次数
        #首先更新taskgraph里的这条链路的路由表
        for i in task_graph[str(task_source)]['out_links']:
            if(int(i[0])==task_dest):
                i[2]=next_partRoute
        #处理partRoute
        partRoute_to_onlineCompute=[]
        partRoute_to_onlineCompute.append(task_source)
        partRoute_to_onlineCompute.append(task_dest)
        partRoute_to_onlineCompute.append(next_partRoute)
        task=onlineTimeline("",num_of_rows)
        task.loadGraphByDict(task_graph,MapResult,fullRouteFromRL,partRoute_to_onlineCompute)
        pendTimes=task.computeTime()
        #根据pendTimes计算reward
        return [next_state_tensor,next_position,next_partRoute],pendTimes,True


#state为[state_tensor,cur_position,partRouteFromRL]，传进来的partRoute的格式是直接的路由表，没有第一位第二位的task
def Environment(state,action,source_position,dest_position,num_of_rows,task_graph,fullRouteFromRL,task_source,task_dest,MapResult):#用于获得next_state，reward，done，除了state和action，剩下的参数都是为了传进onlineCompute
    next_state_tensor=state[0]
    next_position=-1
    next_partRoute=state[2]

    #执行action前
    cur_row=int(state[1]/num_of_rows)
    cur_col=state[1]%num_of_rows
    dest_row=int(dest_position/num_of_rows)
    dest_col=dest_position%num_of_rows
    
    #RL学习之前check一次，就能确保起码能走一步

    #开始执行action
    #state_tensor的四个channel,从0-3以此为N,S,W,E
    if(action==0):#沿x轴走
        if(cur_col<dest_col):
            next_state_tensor[0][3][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'E'])#更新路由表
            cur_col+=1#向East走了一步
        elif(cur_col>dest_col):
            next_state_tensor[0][2][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'W'])#更新路由表
            cur_col-=1#向West走了一步
    elif(action==1):#沿y轴走
        if(cur_row<dest_row):
            next_state_tensor[0][1][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'S'])#更新路由表
            cur_row+=1#向South走了一步
        elif(cur_row>dest_row):
            next_state_tensor[0][0][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'N'])#更新路由表
            cur_row-=1#向North走了一步
    
    next_position=cur_row*num_of_rows+cur_col

    ret_state,ret_reward,done=check_if_Done([next_state_tensor,next_position,next_partRoute],source_position,dest_position,num_of_rows,task_graph,fullRouteFromRL,task_source,task_dest,MapResult)

    if(done==True):
        return ret_state,ret_reward,done
    else:#没有结束，需要计算reward
        #根据XY-routing补全路由表，然后传进onlineCompute
        fullRouteByXY=Get_full_route_by_XY(next_partRoute,source_position,dest_position,num_of_rows)
        #处理参数，传进onlineCompute计算pending次数
        #首先更新taskgraph里的这条链路的路由表
        for i in task_graph[str(task_source)]['out_links']:
            if(int(i[0])==task_dest):
                i[2]=fullRouteByXY
        #处理partRoute
        partRoute_to_onlineCompute=[]
        partRoute_to_onlineCompute.append(task_source)
        partRoute_to_onlineCompute.append(task_dest)
        partRoute_to_onlineCompute.append(next_partRoute)
        task=onlineTimeline("",num_of_rows)
        task.loadGraphByDict(task_graph,MapResult,fullRouteFromRL,partRoute_to_onlineCompute)
        pendTimes=task.computeTime()
        #根据pendTimes计算reward
        return [next_state_tensor,next_position,next_partRoute],pendTimes,False
    











hyperperiod,num_of_tasks,edges,comp_cost=init('./task graph/N4_test.tgff')
adj_matrix,total_needSend,total_needReceive,execution=Get_detailed_data(num_of_tasks,edges,comp_cost)
num_of_rows=4

routeCompute(adj_matrix,num_of_tasks,execution,num_of_rows,1)






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