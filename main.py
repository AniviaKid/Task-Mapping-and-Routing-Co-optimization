import numpy as np
import json
from libs import init,Get_Neighborhood,Get_mapping_reward,Get_rand_computation_ability,Get_detailed_data
import copy
from routing import routeCompute
import time

#M*N 2D-mesh
"""
computation_ability=np.array([
 [1, 1, 2, 1],
 [2, 1, 3, 1],
 [3, 3, 1, 2],
 [1, 3, 2, 2]])
 """
computation_ability=Get_rand_computation_ability(num_of_rows=8)#2的指数级，4/8/16

M=computation_ability.shape[0]
N=computation_ability.shape[1]

#init input
hyperperiod,num_of_tasks,edges,comp_cost=init('./task graph/N12_autocor.tgff')
adj_matrix,total_needSend,total_needReceive,execution=Get_detailed_data(num_of_tasks,edges,comp_cost)



#M*N mesh network, task placed on (i,j) PE
#current solution
#task编号都是从0开始的，跟routing部分不一样，需要适配一下
PEs_task_current_solution=[] #PEs_task_current_solution[i] is a list, means pending tasks of PE(i/N,i%N)
Tasks_position_current_solution={}#key-value: key=task, value=position in mesh


#best solution
PEs_task_best_solution=copy.deepcopy(PEs_task_current_solution)
Tasks_position_best_solution=copy.deepcopy(Tasks_position_current_solution)
best_task_graph={}
Best_reward=-999999999

Tabu_list=[]
Tabu_length=2
Search_radius=1

def Initialization(num_of_tasks): 
    for i in range(0,M*N):
        PEs_task_current_solution.append([])
    for i in range(0,num_of_tasks):
        rand_tmp=np.random.randint(0,M*N)
        PEs_task_current_solution[rand_tmp].append(i)
        Tasks_position_current_solution.update({i:rand_tmp})



def Update_tabu_list(choosen_position, target_position):
    #delete front element in queue
    if(len(Tabu_list)==Tabu_length): 
        Tabu_list.pop(0)

    #insert new tabu element
    s=str(choosen_position)+" "+str(target_position)
    Tabu_list.append(s)




def Iteration(num_of_tasks,radius,num_of_rows): #expand neighborhood, find the fittest, update best solution and tabu list
    """
    print("Current Sol:")
    print(PEs_task_current_solution)
    print(Tasks_position_current_solution)
    print("------------")
    print("Current reward=",Get_mapping_reward(PEs_task_current_solution,computation_ability,M,N))
    print("------------")
    """
    randomly_selected_task=np.random.randint(0,num_of_tasks) #randomly choose a task
    #print("Randomly choose:"," task ",randomly_selected_task)
    current_position=Tasks_position_current_solution[randomly_selected_task]
    neighborhood=Get_Neighborhood(current_position,radius,M,N)
    #print("Full Neighborhood:",neighborhood)
    #print("------------")
    target_position=-1
    tmp_reward=-999999999
    cur_task_graph={}
    for i in neighborhood: #visit all neighborhood and find the fittest one in these neighborhoods
        print("neighborhood:",i)
        flag=False
        for j in Tabu_list:
            source=j.split()[0]
            destination=j.split()[1]
            if(source==current_position and destination==i):
                flag=True
                break
            if(source==i and destination==current_position):
                flag=True
                break
        if(flag):
            #print(i," is baned")
            #print("------------")
            continue
        tmp_solution=copy.deepcopy(PEs_task_current_solution)
        tmp_solution[current_position].remove(randomly_selected_task)
        tmp_solution[i].append(randomly_selected_task)
        reward=Get_mapping_reward(tmp_solution,computation_ability,M,N)
        print("mapping_reward:",reward)

        tmp_mapresults=copy.deepcopy(Tasks_position_current_solution)
        tmp_mapresults.update({randomly_selected_task:i})
        tmp_mapresults1=[-1]#把task编号改成从1开始，然后传给routing部分
        for j in range(0,num_of_tasks):
            tmp_mapresults1.append(tmp_mapresults[j])
        #print("tmp_map_result=",tmp_mapresults1)
        #通过RL计算最佳routing，然后获得routing的reward
        #首先根据MapResult更新execution
        execution_to_routing=copy.deepcopy(execution)
        #print("exe:",execution_to_routing)
        #print("map:",tmp_mapresults1)
        for j in range(1,num_of_tasks+1):
            #print("j=",j)
            execution_to_routing[j]=int(execution_to_routing[j]/computation_ability[int(tmp_mapresults1[j]/num_of_rows)][tmp_mapresults1[j]%num_of_rows])
        routing_reward,ret_task_graph=routeCompute(adj_matrix,num_of_tasks,execution_to_routing,num_of_rows,tmp_mapresults1)
        #print(ret_task_graph)
        print("routing_reward:",routing_reward)
        reward+=routing_reward
        #print("reward=",reward)

        if(reward>tmp_reward):
            tmp_reward=reward
            target_position=i
            cur_task_graph=copy.deepcopy(ret_task_graph)
    
    #update the current solution
    PEs_task_current_solution[current_position].remove(randomly_selected_task)
    PEs_task_current_solution[target_position].append(randomly_selected_task)
    Tasks_position_current_solution.update({randomly_selected_task:target_position})
    #print("target position:",target_position)
    #print("------------")
  
    #update the best solution
    global Best_reward
    if(tmp_reward>Best_reward):
        #print("Update Best sol")
        Best_reward=tmp_reward
        global PEs_task_best_solution
        PEs_task_best_solution=copy.deepcopy(PEs_task_current_solution)
        global Tasks_position_best_solution
        Tasks_position_best_solution=copy.deepcopy(Tasks_position_current_solution)
        global best_task_graph
        best_task_graph=copy.deepcopy(cur_task_graph)


    Update_tabu_list(current_position,target_position)
    #print("Tabu_list:",Tabu_list)
    #print("------------")
        


    

    

        
    
total_start_time=time.time()
Initialization(num_of_tasks)
#print(Tasks_position_current_solution)
for i in range(0,100):
    print("Iteration ",i,":")
    iteration_start_time=time.time()
    Iteration(num_of_tasks,1,N)
    iteration_end_time=time.time()
    print("This iteration costs ",(iteration_end_time-iteration_start_time)/60,"mins")
#print(PEs_task_current_solution)
#print(Get_reward(PEs_task_current_solution))
total_end_time=time.time()
print("It costs ",(total_end_time-total_start_time)/60,"mins")
print("Best reward=",Best_reward)
f = open("./ret.txt", 'w+')
print(Tasks_position_best_solution,file=f)
print(best_task_graph,file=f)
print(computation_ability,file=f)
f.close()
print("done")
