
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
from libs import init,Get_Neighborhood,Get_detailed_data,find_start_task,get_sorted_dict,ActorCritic,Get_full_route_by_XY,Get_reward_by_pendTimes,Environment,check_if_Done,Actor,Critic
from queue import Queue
from OnlineCompute1 import onlineTimeline
import copy












def routeCompute(adj_matrix,num_of_tasks,execution,num_of_rows,MapResult):
    start_task_list=find_start_task(adj_matrix,num_of_tasks)#入度为0的点的集合
    #print("start_task_list=",start_task_list)
    task_graph={}
    fullRouteFromRL=[]#(task_source,task_dest)表示从task_source到task_dest的route全部是由RL计算的
    adj_matrix_tmp=copy.deepcopy(adj_matrix)
    

    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    visited={}


    for start_task in start_task_list:
        #print("start at ",start_task)
        q=Queue(maxsize=0)
        q.put(start_task)
        while(q.empty()!=True):#BFS
            u=q.get()
            #print("we get u=",u,"now queue is ",q.queue)
            adj_u={}#task_dest - send_size，用于排序寻找用时最短的任务
            for i in range(1,num_of_tasks+1):
                if(adj_matrix_tmp[u][i]!=0):
                    adj_u[i]=adj_matrix_tmp[u][i]#加入待处理队列
                    adj_matrix_tmp[u][i]=0
            adj_u=sorted(adj_u.items(), key=lambda x:x[1])#最短任务优先
            #print("u=",u,"adj_u:",adj_u)
            for i in adj_u:
                if(u in visited.keys() and visited[u]==i[0]):
                    #print("this edge ",u,"to",i[0],"has visited")
                    continue
                #print("visit edge",u,"to",i[0],"size=",i[1])
                q.put(i[0])
                #print(i[0],' is pushed')
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
                task_graph=get_sorted_dict(task_graph)
                
                #开始为边u->i[0]计算路由
                #state为[state_tensor,cur_position,partRouteFromRL]，传进来的partRoute的格式是直接的路由表，没有第一位第二位的task
                #state_tensor的四个channel,从0-3以此为N,S,W,E
                state_tensor=torch.Tensor(np.zeros((1,4,num_of_rows*num_of_rows),dtype=np.int)).to(device)
                state=[state_tensor,MapResult[u],[]]
                #确保当前的state(map后的位置)不是end state，至少能执行一次action
                tmp_state,_,tmp_done=check_if_Done(state,source_position=MapResult[u],dest_position=MapResult[i[0]],num_of_rows=num_of_rows,task_graph=task_graph,fullRouteFromRL=fullRouteFromRL,task_source=u,task_dest=i[0],MapResult=MapResult)
                if(tmp_done):#这两个task的位置无需计算route，直接结束，需要更新fullRoute和taskgraph
                    #首先更新taskgraph
                    for j in range(0,len(task_graph[str(u)]['out_links'])):
                        if(int(task_graph[str(u)]['out_links'][j][0])==int(i[0])):
                            task_graph[str(u)]['out_links'][j][2]=tmp_state[2]
                    #更新fullRoute
                    fullRouteFromRL.append((u,i[0]))
                    continue
                #开始RL
                actor=Actor(4,num_of_rows*num_of_rows,2).to(device)
                critic=Critic(4,num_of_rows*num_of_rows).to(device)
                adam_actor=optim.Adam(actor.parameters(),lr=1e-3)
                adam_critic=optim.Adam(critic.parameters(), lr=1e-3)
                gamma=0.99
                episode_rewards = []
                best_Route=[]
                best_reward=-99999
                for kkk in range(100):
                    #print("iteration in RL is ",kkk,"edge=",u,"->",i[0])
                    done=False
                    total_reward=0
                    state_tensor=torch.Tensor(np.zeros((1,4,num_of_rows*num_of_rows),dtype=np.int))
                    state=[state_tensor,MapResult[u],[]]

                    while not done:
                        state[0]=state[0].to(device)
                        probs=actor(state[0])
                        dist = torch.distributions.Categorical(probs=probs)
                        action = dist.sample()

                        next_state,reward,done=Environment(state,int(action),source_position=MapResult[u],dest_position=MapResult[i[0]],num_of_rows=num_of_rows,task_graph=task_graph,fullRouteFromRL=fullRouteFromRL,task_source=u,task_dest=i[0],MapResult=MapResult)
                        d=0
                        next_state[0]=next_state[0].to(device)
                        if(done):
                            d=1
                        advantage=reward+(1-d)*gamma*critic(next_state[0])-critic(state[0])


                        total_reward+=reward
                        state=next_state

                        critic_loss = advantage.pow(2).mean()
                        adam_critic.zero_grad()
                        critic_loss.backward()
                        adam_critic.step()

                        actor_loss = -dist.log_prob(action)*advantage.detach()
                        adam_actor.zero_grad()
                        actor_loss.backward()
                        adam_actor.step()
                    
                    if(total_reward>best_reward):
                        best_reward=total_reward
                        best_Route=state[2]
                    episode_rewards.append(total_reward)
                #print(episode_rewards)
                for j in range(0,len(task_graph[str(u)]['out_links'])):
                    if(int(task_graph[str(u)]['out_links'][j][0])==int(i[0])):
                        task_graph[str(u)]['out_links'][j][2]=best_Route
                #print(get_sorted_dict(task_graph))
                fullRouteFromRL.append((u,i[0]))
                visited.update({u:i[0]})


                

    #print(get_sorted_dict(task_graph))
    task_graph=get_sorted_dict(task_graph)
    task=onlineTimeline("",num_of_rows)
    #print(task_graph)
    task.loadGraphByDict(task_graph,MapResult,fullRouteFromRL,[-1,-1],num_of_tasks)
    pendTimes=task.computeTime()
    #return Get_reward_by_pendTimes(pendTimes),task_graph
    return pendTimes,task_graph


class link_item():#可以根据在list中的下标索引到它连接的是哪两个PE
    def __init__(self):
        #记录这个link的timeline，list中的每个元素是list，形式为[task_source,task_dest,start_time,end_time]
        self.timeline=[]
    

#adj_matrix里task的编号是从1开始的，我的edge_set编号也是从1开始的
def improved_routeCompute(adj_matrix,num_of_tasks,execution,num_of_rows,MapResult):
    edge_set={}#'2,3':{'transmission':10,'used_link':[]}
    link_set=[]
    receiveMatrix = [-1]
    total_link_num=(num_of_rows-1+num_of_rows)*(num_of_rows-1)+num_of_rows-1
    for i in range(0,total_link_num):
        tmp=link_item()
        link_set.append(tmp)
    for i in range(1,num_of_tasks+1):#初始化edge_set
        for j in range(1,num_of_tasks+1):
            if(adj_matrix[i][j]!=0):
                tmp_key=str(i)+','+str(j)
                edge_set.update({tmp_key:{'transmission':adj_matrix[i][j],'used_link':[]}})
    for i in range(1,num_of_tasks+1):#初始化receive_matrix，这里遍历的是每一列
        total_receive_for_i=0
        for j in range(1,num_of_tasks+1):
            total_receive_for_i+=adj_matrix[j][i]
        receiveMatrix.append(total_receive_for_i)
    
            



    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    contention=0
    edge_queue=[]#每个item为( 'task_source,task_dest' , end time of task_source )，如('1,2',20)
    #添加一开始就能执行的边
    for i in range(1,num_of_tasks+1):
        if(receiveMatrix[i]==0):#这个task可以立刻执行，然后开始传输
            for j in range(1,num_of_tasks):
                if(adj_matrix[i][j]!=0):
                    tmp=(str(i)+','+j,execution[i])
                    edge_queue.append(tmp)
    edge_queue.sort(key=lambda x: x[1])#按照task_source的结束时间排序
    

    #队列不空时，取队首的边来执行，将这条边以及link_set传入RL模型，RL模型还是step-by-step来计算每一步应该怎么走，但是计算reward的时候，就只需要检查在这些由RL模型计算出的link上，发生的最长的争用，然后将它作为这一步的reward就好了
    #而检查指定的link的争用情况，只需要检查link在[task_source的结束时间,task_source的结束时间+transmission]时间段是否可用，如果可用的话就占用这个时间段的这些link，如果不可用，则等待时间T，直到[task_source的结束时间+T,task_source的结束时间+T+transmission]，而这个时间T就是这条link上的contention
    #RL模型训练结束后，记录在训练过程中出现的最好的route，作为这条edge的route
                
    """

    start_task_list=find_start_task(adj_matrix,num_of_tasks)#入度为0的点的集合
    #print("start_task_list=",start_task_list)
    task_graph={}
    fullRouteFromRL=[]#(task_source,task_dest)表示从task_source到task_dest的route全部是由RL计算的
    adj_matrix_tmp=copy.deepcopy(adj_matrix)
    

    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    visited={}


    for start_task in start_task_list:
        #print("start at ",start_task)
        q=Queue(maxsize=0)
        q.put(start_task)
        while(q.empty()!=True):#BFS
            u=q.get()
            #print("we get u=",u,"now queue is ",q.queue)
            adj_u={}#task_dest - send_size，用于排序寻找用时最短的任务
            for i in range(1,num_of_tasks+1):
                if(adj_matrix_tmp[u][i]!=0):
                    adj_u[i]=adj_matrix_tmp[u][i]#加入待处理队列
                    adj_matrix_tmp[u][i]=0
            adj_u=sorted(adj_u.items(), key=lambda x:x[1])#最短任务优先
            #print("u=",u,"adj_u:",adj_u)
            for i in adj_u:
                if(u in visited.keys() and visited[u]==i[0]):
                    #print("this edge ",u,"to",i[0],"has visited")
                    continue
                #print("visit edge",u,"to",i[0],"size=",i[1])
                q.put(i[0])
                #print(i[0],' is pushed')
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
                task_graph=get_sorted_dict(task_graph)
                
                #开始为边u->i[0]计算路由
                #state为[state_tensor,cur_position,partRouteFromRL]，传进来的partRoute的格式是直接的路由表，没有第一位第二位的task
                #state_tensor的四个channel,从0-3以此为N,S,W,E
                state_tensor=torch.Tensor(np.zeros((1,4,num_of_rows*num_of_rows),dtype=np.int)).to(device)
                state=[state_tensor,MapResult[u],[]]
                #确保当前的state(map后的位置)不是end state，至少能执行一次action
                tmp_state,_,tmp_done=check_if_Done(state,source_position=MapResult[u],dest_position=MapResult[i[0]],num_of_rows=num_of_rows,task_graph=task_graph,fullRouteFromRL=fullRouteFromRL,task_source=u,task_dest=i[0],MapResult=MapResult)
                if(tmp_done):#这两个task的位置无需计算route，直接结束，需要更新fullRoute和taskgraph
                    #首先更新taskgraph
                    for j in range(0,len(task_graph[str(u)]['out_links'])):
                        if(int(task_graph[str(u)]['out_links'][j][0])==int(i[0])):
                            task_graph[str(u)]['out_links'][j][2]=tmp_state[2]
                    #更新fullRoute
                    fullRouteFromRL.append((u,i[0]))
                    continue
                #开始RL
                actor=Actor(4,num_of_rows*num_of_rows,2).to(device)
                critic=Critic(4,num_of_rows*num_of_rows).to(device)
                adam_actor=optim.Adam(actor.parameters(),lr=1e-3)
                adam_critic=optim.Adam(critic.parameters(), lr=1e-3)
                gamma=0.99
                episode_rewards = []
                best_Route=[]
                best_reward=-99999
                for kkk in range(100):
                    #print("iteration in RL is ",kkk,"edge=",u,"->",i[0])
                    done=False
                    total_reward=0
                    state_tensor=torch.Tensor(np.zeros((1,4,num_of_rows*num_of_rows),dtype=np.int))
                    state=[state_tensor,MapResult[u],[]]

                    while not done:
                        state[0]=state[0].to(device)
                        probs=actor(state[0])
                        dist = torch.distributions.Categorical(probs=probs)
                        action = dist.sample()

                        next_state,reward,done=Environment(state,int(action),source_position=MapResult[u],dest_position=MapResult[i[0]],num_of_rows=num_of_rows,task_graph=task_graph,fullRouteFromRL=fullRouteFromRL,task_source=u,task_dest=i[0],MapResult=MapResult)
                        d=0
                        next_state[0]=next_state[0].to(device)
                        if(done):
                            d=1
                        advantage=reward+(1-d)*gamma*critic(next_state[0])-critic(state[0])


                        total_reward+=reward
                        state=next_state

                        critic_loss = advantage.pow(2).mean()
                        adam_critic.zero_grad()
                        critic_loss.backward()
                        adam_critic.step()

                        actor_loss = -dist.log_prob(action)*advantage.detach()
                        adam_actor.zero_grad()
                        actor_loss.backward()
                        adam_actor.step()
                    
                    if(total_reward>best_reward):
                        best_reward=total_reward
                        best_Route=state[2]
                    episode_rewards.append(total_reward)
                #print(episode_rewards)
                for j in range(0,len(task_graph[str(u)]['out_links'])):
                    if(int(task_graph[str(u)]['out_links'][j][0])==int(i[0])):
                        task_graph[str(u)]['out_links'][j][2]=best_Route
                #print(get_sorted_dict(task_graph))
                fullRouteFromRL.append((u,i[0]))
                visited.update({u:i[0]})


                

    #print(get_sorted_dict(task_graph))
    task_graph=get_sorted_dict(task_graph)
    task=onlineTimeline("",num_of_rows)
    #print(task_graph)
    task.loadGraphByDict(task_graph,MapResult,fullRouteFromRL,[-1,-1],num_of_tasks)
    pendTimes=task.computeTime()
    #return Get_reward_by_pendTimes(pendTimes),task_graph
    return pendTimes,task_graph
    """






if __name__ == '__main__':

    hyperperiod,num_of_tasks,edges,comp_cost=init('./task graph/N12_autocor_AIR1.tgff')
    adj_matrix,total_needSend,total_needReceive,execution=Get_detailed_data(num_of_tasks,edges,comp_cost)
    #print(adj_matrix)
    num_of_rows=4
    MapResults=[-1,4,1,10,3]

    print(improved_routeCompute(adj_matrix,num_of_tasks,execution,num_of_rows,MapResults))






