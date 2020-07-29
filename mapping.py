import numpy as np
import json

#M*N 2D-mesh
M=8 
N=8


#M*N mesh network, task placed on (i,j) PE, task's start/end time on (i,j) PE
#current solution
PEs_task_current_solution=np.full(shape=(M,N),fill_value=-1)
PEs_start_time_current_solution=np.zeros((M,N))
PEs_end_time_current_solution=np.zeros((M,N))
Tasks_position_current_solution={}#key-value: key=task, value=position in mesh


#best solution
PEs_task_best_solution=PEs_task_current_solution.copy()
PEs_start_time_best_solution=PEs_start_time_current_solution.copy()
PEs_end_time_best_solution=PEs_end_time_current_solution.copy()
Best_reward=0

Tabu_list={}
Tabu_length=10
Search_radius=1

def Initialization(task_list): #element in list is str
    for i in task_list:
            while(1):
                rand_tmp=np.random.randint(0,M*N)
                if(PEs_task_current_solution[int(rand_tmp/N)][rand_tmp%N]==-1): #find an empty position
                    PEs_task_current_solution[int(rand_tmp/N)][rand_tmp%N]=int(i)
                    Tasks_position_current_solution.update({i:rand_tmp})
                    break

def Get_Neighborhood(position,radius): #return a list which consists of positions around input position with radius=r
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


def Iteration(task_list,radius): #expand neighborhood, find the fittest, update best solution and tabu list
    random_choose=Tasks_position_current_solution[str(np.random.randint(0,len(task_list)))]
    neighborhood=Get_Neighborhood(random_choose,radius)
    tmp_reward=-999999999
    target_position=-1
    for i in neighborhood: #visit all neighborhood and find the fittest one in these neighborhoods
        if(random_choose in Tabu_list.keys() and i in Tabu_list.keys()):
            continue
        reward=Get_reward(PEs_task_current_solution,random_choose,i)#待完成
        if(reward>tmp_reward):
            tmp_reward=reward
            target_position=i
    
    #update the current solution
    row_chosen=int(random_choose)/N
    col_chosen=random_choose%N
    row_target=int(target_position)/N
    col_target=target_position%N
    tmp=PEs_task_current_solution[row_chosen][col_chosen]
    PEs_task_current_solution[row_chosen][col_chosen]=PEs_task_current_solution[row_target][col_target]
    PEs_task_current_solution[row_target][col_target]=tmp

        
    #update the best solution
    if(tmp_reward>Best_reward):
        Best_reward=tmp_reward
        PEs_task_best_solution=PEs_task_current_solution.copy()

    Update_tabu_list(random_choose,target_position)
        




    






def Get_reward(PEs_task_current_solution, chosen_position, target_position):
    return 0
    

def tabu_search(json_file_name):
    with open(json_file_name,'r') as load_json:
        json_file=json.load(load_json)

        #initialization
        Initialization(list(json_file.keys()))
        print(Tasks_position_current_solution)
        Expand_Neighborhood(list(json_file.keys()),1)
    
if __name__ == '__main__':
    tabu_search("./1_8_Autocor_basic.json")