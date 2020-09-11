
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
from libs import init,Get_Neighborhood,Get_mapping_reward,Get_detailed_data,find_start_task,get_sorted_dict,ActorCritic,Get_full_route_by_XY,Get_reward_by_pendTimes,Environment,check_if_Done,Actor,Critic
from queue import Queue
from OnlineCompute1 import onlineTimeline












def routeCompute(adj_matrix,num_of_tasks,execution,num_of_rows,MapResult):
    start_task_list=find_start_task(adj_matrix,num_of_tasks)#入度为0的点的集合
    task_graph={}
    fullRouteFromRL={}#key-value表示从task_key到task_value的route全部是由RL计算的
    #partRouteFromRL=[]#它的元素就是json里的route格式，如[0, "S"]，当前位置+下一步移动方向

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
                #state为[state_tensor,cur_position,partRouteFromRL]，传进来的partRoute的格式是直接的路由表，没有第一位第二位的task
                #state_tensor的四个channel,从0-3以此为N,S,W,E
                state_tensor=torch.Tensor(np.zeros((1,4,num_of_rows*num_of_rows),dtype=np.int))
                state=[state_tensor,MapResult[u],[]]
                #确保当前的state(map后的位置)不是end state，至少能执行一次action
                tmp_state,_,tmp_done=check_if_Done(state,source_position=MapResult[u],dest_position=MapResult[i[0]],num_of_rows=num_of_rows,task_graph=task_graph,fullRouteFromRL=fullRouteFromRL,task_source=u,task_dest=i[0],MapResult=MapResult)
                if(tmp_done):#这两个task的位置无需计算route，直接结束，需要更新fullRoute和taskgraph
                    #首先更新taskgraph
                    for j in task_graph[str(u)]['out_links']:
                        if(int(j[0])==int(i[0])):
                            j[2]=tmp_state[2]
                    #更新fullRoute
                    fullRouteFromRL.update({u:i[0]})
                #开始RL
                actor=Actor(4,num_of_rows*num_of_rows,2)
                critic=Critic(4,num_of_rows*num_of_rows)
                adam_actor=optim.Adam(actor.parameters(),lr=1e-3)
                adam_critic=optim.Adam(critic.parameters(), lr=1e-3)
                gamma=0.99
                episode_rewards = []
                for _ in range(500):
                    done=False
                    total_reward=0
                    state_tensor=torch.Tensor(np.zeros((1,4,num_of_rows*num_of_rows),dtype=np.int))
                    state=[state_tensor,MapResult[u],[]]

                    while not done:
                        probs=actor(state[0])
                        dist = torch.distributions.Categorical(probs=probs)
                        action = dist.sample()

                        next_state,reward,done=Environment(state,int(action),source_position=MapResult[u],dest_position=MapResult[i[0]],num_of_rows=num_of_rows,task_graph=task_graph,fullRouteFromRL=fullRouteFromRL,task_source=u,task_dest=i[0],MapResult=MapResult)
                        d=0
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
                        
                    episode_rewards.append(total_reward)

                

    print(get_sorted_dict(task_graph))


    











hyperperiod,num_of_tasks,edges,comp_cost=init('./task graph/N4_test.tgff')
adj_matrix,total_needSend,total_needReceive,execution=Get_detailed_data(num_of_tasks,edges,comp_cost)
num_of_rows=4

routeCompute(adj_matrix,num_of_tasks,execution,num_of_rows,1)






