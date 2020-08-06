import numpy as np
import json
from libs import init,Get_Neighborhood,Get_reward
import copy

#M*N 2D-mesh
computation_ability=np.array([
 [1, 1, 2, 1],
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
PEs_task_best_solution=copy.deepcopy(PEs_task_current_solution)
Best_reward=0

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




def Iteration(num_of_tasks,radius): #expand neighborhood, find the fittest, update best solution and tabu list
    print("Current Sol:")
    print(PEs_task_current_solution)
    print(Tasks_position_current_solution)
    print("------------")
    print("Current reward=",Get_reward(PEs_task_current_solution,computation_ability,M,N))
    print("------------")
    randomly_selected_task=np.random.randint(0,num_of_tasks) #randomly choose a task
    print("Randomly choose:"," task ",randomly_selected_task)
    current_position=Tasks_position_current_solution[randomly_selected_task]
    neighborhood=Get_Neighborhood(current_position,radius,M,N)
    print("Neighborhood:",neighborhood)
    print("------------")
    target_position=-1
    tmp_reward=-9999
    for i in neighborhood: #visit all neighborhood and find the fittest one in these neighborhoods
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
            print(i," is baned")
            print("------------")
            continue
        tmp_solution=copy.deepcopy(PEs_task_current_solution)
        tmp_solution[current_position].remove(randomly_selected_task)
        tmp_solution[i].append(randomly_selected_task)
        reward=Get_reward(tmp_solution,computation_ability,M,N)
        if(reward>tmp_reward):
            tmp_reward=reward
            target_position=i
    
    #update the current solution
    PEs_task_current_solution[current_position].remove(randomly_selected_task)
    PEs_task_current_solution[target_position].append(randomly_selected_task)
    Tasks_position_current_solution.update({randomly_selected_task:target_position})
    print("target position:",target_position)
    print("------------")
  
    #update the best solution
    global Best_reward
    if(tmp_reward>Best_reward):
        print("Update Best sol")
        Best_reward=tmp_reward
        PEs_task_best_solution=copy.deepcopy(PEs_task_current_solution)

    Update_tabu_list(current_position,target_position)
    print("Tabu_list:",Tabu_list)
    print("------------")
        


    


    

        
    
if __name__ == '__main__':
    Initialization(num_of_tasks)
    for i in range(0,50):
        print("Iteration ",i,":")
        Iteration(num_of_tasks,1)
    #print(PEs_task_current_solution)
    #print(Get_reward(PEs_task_current_solution))
    print("Best reward=",Best_reward)
    print(PEs_task_best_solution)