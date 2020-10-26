import numpy as np
import json
from libs import init,Get_Neighborhood,Get_mapping_exe_time,Get_detailed_data,Get_rand_computation_ability2,CVB_method
import copy
from routing import routeCompute,improved_routeCompute
import time

#M*N 2D-mesh
"""
computation_ability=np.array([
 [1, 1, 2, 1],
 [2, 1, 3, 1],
 [3, 3, 1, 2],
 [1, 3, 2, 2]])
 """
#computation_ability=Get_rand_computation_ability2(num_of_rows=8)#2的指数级，4/8/16


#M=computation_ability.shape[0]
#N=computation_ability.shape[1]
M=8#num_of_rows
N=8#num_of_rows

#init input
hyperperiod,num_of_tasks,edges,comp_cost=init('./task graph/N12_autocor.tgff')
adj_matrix,total_needSend,total_needReceive,execution=Get_detailed_data(num_of_tasks,edges,comp_cost)

#computation_ability=CVB_method(execution=execution[1:],V_machine=0.5,num_of_rows=N)
computation_ability=np.array([[233,106,355,134,245,309,315,566,292,419,440,394,294,193
,345,165,347,362,450,316,563,633,398,327,277,229,258,523
,479,97,199,822,212,130,207,391,541,161,325,126,292,814
,410,278,385,233,608,244,341,252,353,343,468,388,545,229
,366,201,182,790,129,421,499,166]
,[1101,1470,640,922,729,2334,614,3959,2315,4217,527,3384,1909,694
,1075,1361,1344,2049,3241,1818,1616,1843,1047,784,1896,5197,2403,814
,2769,1815,522,2038,2087,637,3362,2207,895,1340,1383,1610,1439,1409
,3189,5185,1309,3959,1073,3006,715,1285,1289,1348,2011,2163,1543,1229
,1635,3860,1480,4143,1725,1048,1015,2709]
,[579,352,1074,591,1319,297,504,328,980,432,956,306,571,273
,410,1282,627,959,493,431,653,686,909,494,270,700,683,1488
,510,793,786,555,294,344,469,312,972,167,569,1509,1349,972
,284,702,245,473,417,325,375,298,419,632,986,778,1087,323
,1334,657,1147,541,182,692,296,463]
,[31,91,53,16,40,60,73,51,45,33,9,148,45,130
,101,64,35,98,59,39,19,46,78,71,56,46,61,62
,61,75,129,70,106,24,113,17,52,14,54,117,77,79
,68,32,48,113,70,113,77,74,94,89,153,38,14,51
,81,104,35,42,52,48,101,68]
,[37,29,51,43,41,35,66,26,82,68,46,47,84,22
,83,36,41,54,12,69,24,33,66,43,38,86,73,30
,96,28,90,50,33,29,44,84,75,28,46,54,29,28
,32,52,37,43,27,57,63,41,18,115,40,9,33,22
,34,21,64,32,46,63,77,46]
,[574,405,813,563,632,562,1025,284,427,434,681,294,779,581
,207,404,630,313,291,494,1350,795,217,754,369,753,751,512
,813,1687,300,523,752,131,504,849,325,543,682,534,757,1626
,221,209,358,447,249,716,381,781,424,323,214,298,517,514
,915,605,626,1087,218,301,605,144]
,[759,518,677,428,507,597,569,526,536,247,653,441,923,702
,308,219,620,317,361,297,495,806,741,642,709,983,499,742
,549,344,475,194,175,318,496,846,193,659,364,371,1104,321
,405,527,314,242,505,913,124,161,377,698,494,929,370,275
,207,689,523,1001,856,441,321,372]
,[137,431,622,881,698,725,187,399,172,573,541,226,539,303
,615,405,181,1241,642,205,761,693,570,367,818,380,1470,566
,237,934,610,222,367,496,337,549,1023,309,731,539,510,307
,150,547,387,689,572,776,1016,743,315,104,512,463,456,662
,405,508,103,361,776,125,543,242]
,[576,530,966,137,472,171,484,1296,712,379,385,236,223,833
,428,575,1066,614,789,626,514,570,1274,724,1017,660,389,458
,453,409,854,307,910,360,312,995,516,1118,939,532,625,219
,454,239,677,784,255,714,175,620,849,373,557,239,395,726
,136,453,312,287,703,403,720,953]
,[436,366,973,169,271,375,634,604,456,933,607,93,718,683
,574,431,1023,846,169,557,786,334,443,542,966,572,434,952
,489,1123,441,466,323,536,652,451,366,851,190,488,519,581
,667,825,631,491,295,427,890,627,229,790,217,372,412,846
,850,609,787,459,237,393,245,1140]
,[96,439,544,192,485,399,388,432,76,556,682,333,421,315
,379,199,177,589,234,554,642,389,199,1049,573,703,343,293
,552,492,252,971,321,260,318,785,992,84,788,588,601,308
,449,405,494,324,538,603,846,199,190,688,259,106,361,517
,568,670,365,282,636,483,255,306]
,[727,571,381,466,662,680,318,373,720,418,1202,447,624,384
,308,449,191,454,590,603,431,93,348,405,595,747,421,519
,267,219,313,982,610,434,627,145,287,752,515,203,269,173
,204,406,481,745,323,395,390,742,635,376,580,1318,174,454
,304,256,370,243,707,551,453,244]])



#M*N mesh network, task placed on (i,j) PE
#current solution
#task编号都是从0开始的，跟routing部分不一样，需要适配一下
PEs_task_current_solution=[] #PEs_task_current_solution[i] is a list, means pending tasks of PE(i/N,i%N)
Tasks_position_current_solution={}#key-value: key=task, value=position in mesh


#best solution
PEs_task_best_solution=copy.deepcopy(PEs_task_current_solution)
Tasks_position_best_solution=copy.deepcopy(Tasks_position_current_solution)
best_task_graph={}
Best_reward=999999999

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
    randomly_selected_task=np.random.randint(0,num_of_tasks) #randomly choose a task
    current_position=Tasks_position_current_solution[randomly_selected_task]
    neighborhood=Get_Neighborhood(current_position,radius,M,N)
    target_position=-1
    tmp_reward=999999999
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

        tmp_mapresults=copy.deepcopy(Tasks_position_current_solution)
        tmp_mapresults.update({randomly_selected_task:i})

        mapping_exe_time=Get_mapping_exe_time(PEs_task_current_solution=tmp_solution,Tasks_position_current_solution=tmp_mapresults,computation_ability=computation_ability,num_of_rows=num_of_rows,execution=execution)


        tmp_mapresults1=[-1]#把task编号改成从1开始，然后传给routing部分
        for j in range(0,num_of_tasks):
            tmp_mapresults1.append(tmp_mapresults[j])
        #通过RL计算最佳routing，然后获得routing的reward
        #首先根据MapResult更新execution
        execution_to_routing=copy.deepcopy(execution)
        for j in range(1,num_of_tasks+1):
            #execution_to_routing[j]=int(execution_to_routing[j]/computation_ability[int(tmp_mapresults1[j]/num_of_rows)][tmp_mapresults1[j]%num_of_rows])
            execution_to_routing[j]=computation_ability[j-1][tmp_mapresults1[j]]
        #计算路由
        #pendTimes,ret_task_graph=routeCompute(adj_matrix,num_of_tasks,execution_to_routing,num_of_rows,tmp_mapresults1)
        pendTimes,ret_task_graph=improved_routeCompute(adj_matrix,num_of_tasks,execution_to_routing,num_of_rows,tmp_mapresults1)
        total=mapping_exe_time+pendTimes#这个值应该是越小越好，第一个值小表示mapping结果匹配到的PE计算能力强，第二个值小表示routing中争用少

        if(total<tmp_reward):
            tmp_reward=total
            target_position=i
            cur_task_graph=copy.deepcopy(ret_task_graph)
    
    #update the current solution
    PEs_task_current_solution[current_position].remove(randomly_selected_task)
    PEs_task_current_solution[target_position].append(randomly_selected_task)
    Tasks_position_current_solution.update({randomly_selected_task:target_position})
    print("reward in this iteration is ",tmp_reward)
  
    #update the best solution
    global Best_reward
    if(tmp_reward<Best_reward):#越小越好，详见total=mapping_exe_time+pendTimes处的注释
        #print("Update Best sol")
        Best_reward=tmp_reward
        global PEs_task_best_solution
        PEs_task_best_solution=copy.deepcopy(PEs_task_current_solution)
        global Tasks_position_best_solution
        Tasks_position_best_solution=copy.deepcopy(Tasks_position_current_solution)
        global best_task_graph
        best_task_graph=copy.deepcopy(cur_task_graph)


    Update_tabu_list(current_position,target_position)
        


    

    

        
    
total_start_time=time.time()
Initialization(num_of_tasks)
#print(Tasks_position_current_solution)
for i in range(0,150):
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
#print(computation_ability,file=f)
print("[",file=f)
for i in computation_ability:
    print("[",end="",file=f)
    for j in i:
        print(j,end=",",file=f)
    print("]",file=f)
print("]",file=f)
f.close()
print("done")
