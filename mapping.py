import numpy as np
import json
from create_input import init

#M*N 2D-mesh
computation_ability=np.array([[1, 1, 2, 1],
 [2, 1, 3, 1],
 [3, 3, 1, 2],
 [1, 3, 2, 2]])

M=computation_ability.shape[0]
N=computation_ability.shape[1]

#init input
hyperperiod,num_of_tasks,edges,comp_cost=init('./task graph/N4_test.tgff')


#M*N mesh network, task placed on (i,j) PE
#current solution
PEs_task_current_solution=[] #PEs_task_current_solution[i] is a list, means pending tasks of PE(i/N,i%N)
Tasks_position_current_solution={}#key-value: key=task, value=position in mesh


#best solution
PEs_task_best_solution=PEs_task_current_solution.copy()
Best_reward=0

Tabu_list={}
Tabu_length=10
Search_radius=1

def Initialization(num_of_tasks): 
    for i in range(0,M*N):
        PEs_task_current_solution.append([])
    for i in range(0,num_of_tasks):
        rand_tmp=np.random.randint(0,M*N)
        PEs_task_current_solution[rand_tmp].append(i)
        Tasks_position_current_solution.update({i:rand_tmp})

def Get_Neighborhood(position,radius,M,N): #return a list which consists of positions around input position with radius=r
    row=int(position/N)
    col=position%N
    neighborhood=[]
    for i in range(row-radius,row+radius+1):
        if(i>=0 and i<M):
            for j in range(col-radius,col+radius+1):
                if(j>=0 and (i!=row or j!=col) and j<N):
                    neighborhood.append(i*N+j)
    return neighborhood

def Update_tabu_list(choosen_position, target_position):
    #delete front element in queue
    if(len(Tabu_list)==2*Tabu_length): 
        cnt=0
        del_keys=[]
        for i in Tabu_list.keys():
            del_keys.append(i)
            cnt+=1
            if(cnt==2):
                break

        for i in del_keys:
            Tabu_list.pop(i)

    #insert new tabu element
    Tabu_list.update({choosen_position:target_position})
    Tabu_list.update({target_position:choosen_position})

def Get_reward(PEs_task_current_solution):
    ret=0
    for i in range(0,len(PEs_task_current_solution)):
        if(len(PEs_task_current_solution[i])): #this PE has tasks
            ret+=computation_ability[int(i/N)][i%N]
    return ret


def Iteration(num_of_tasks,radius): #expand neighborhood, find the fittest, update best solution and tabu list
    randomly_selected_task=np.random.randint(0,num_of_tasks) #randomly choose a task
    current_position=Tasks_position_current_solution[randomly_selected_task]
    neighborhood=Get_Neighborhood(current_position,radius)
    tmp_reward=-999999999
    target_position=-1
    for i in neighborhood: #visit all neighborhood and find the fittest one in these neighborhoods
        if(current_position in Tabu_list.keys() and i in Tabu_list.keys()):
            continue
        reward=Get_reward(PEs_task_current_solution)#待完成
        if(reward>tmp_reward):
            tmp_reward=reward
            target_position=i
    
    #update the current solution
    PEs_task_current_solution[current_position].remove(randomly_selected_task)
    PEs_task_current_solution[target_position].append(randomly_selected_task)
    Tasks_position_current_solution.update({randomly_selected_task:target_position})
  
    #update the best solution
    if(tmp_reward>Best_reward):
        Best_reward=tmp_reward
        PEs_task_best_solution=PEs_task_current_solution.copy()

    Update_tabu_list(current_position,target_position)
        


    


    

        
    
if __name__ == '__main__':
    Initialization(num_of_tasks)
    print(PEs_task_current_solution)
    print(Get_reward(PEs_task_current_solution))