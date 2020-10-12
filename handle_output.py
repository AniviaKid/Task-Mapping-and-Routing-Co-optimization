from libs import init,Get_detailed_data
import numpy as np
import json


MapResults={0: 1, 1: 36, 2: 31, 3: 19, 4: 60, 5: 44, 6: 11, 7: 6, 8: 45, 9: 28, 10: 8, 11: 42}
task_graph={'1': {'out_links': [['2', 93, [[1, 'E'], [2, 'E'], [3, 'E'], [4, 'S'], [12, 'S'], [20, 'S'], [28, 'S']], 0, 0, -1]]}, '2': {'out_links': [['4', 93, [[36, 'N'], [28, 'W'], [27, 'N']], 0, 0, -1], ['5', 93, [[36, 'S'], [44, 'S'], [52, 'S']], 0, 0, -1], ['6', 93, [[36, 'S']], 0, 0, -1], ['7', 93, [[36, 'W'], [35, 'N'], [27, 'N'], [19, 'N']], 0, 0, -1], ['8', 93, [[36, 'E'], [37, 'E'], [38, 'N'], [30, 'N'], [22, 'N'], [14, 'N']], 0, 0, -1], ['9', 93, [[36, 'E'], [37, 'S']], 0, 0, -1], ['10', 93, [[36, 'N']], 0, 0, -1], ['11', 93, [[36, 'W'], [35, 'W'], [34, 'N'], [26, 'W'], [25, 'W'], [24, 'N'], [16, 'N']], 0, 0, -1]]}, '4': {'out_links': [['3', 93, [[19, 'E'], [20, 'S'], [28, 'E'], [29, 'E'], [30, 'E']], 0, 0, -1]]}, '5': {'out_links': [['3', 93, [[60, 'E'], [61, 'E'], [62, 'N'], [54, 'N'], [46, 'E'], [47, 'N'], [39, 'N']], 0, 0, -1]]}, '6': {'out_links': [['3', 93, [[44, 'E'], [45, 'E'], [46, 'N'], [38, 'E'], [39, 'N']], 0, 0, -1]]}, '7': {'out_links': [['3', 93, [[11, 'E'], [12, 'E'], [13, 'S'], [21, 'E'], [22, 'E'], [23, 'S']], 0, 0, -1]]}, '8': {'out_links': [['3', 93, [[6, 'S'], [14, 'E'], [15, 'S'], [23, 'S']], 0, 0, -1]]}, '9': {'out_links': [['3', 93, [[45, 'N'], [37, 'E'], [38, 'E'], [39, 'N']], 0, 0, -1]]}, '10': {'out_links': [['3', 93, [[28, 'E'], [29, 'E'], [30, 'E']], 0, 0, -1]]}, '11': {'out_links': [['3', 93, [[8, 'E'], [9, 'S'], [17, 'E'], [18, 'S'], [26, 'E'], [27, 'E'], [28, 'E'], [29, 'E'], [30, 'E']], 0, 0, -1]]}, '3': {'out_links': [['12', 93, [[31, 'W'], [30, 'S'], [38, 'W'], [37, 'S'], [45, 'W'], [44, 'W'], [43, 'W']], 0, 0, -1]]}, '12': {'out_links': []}}

computation_ability=np.array([[0.5,1,0.5,1.5,0.5,0.5,1.5,1]
,[1,0.5,1,1.5,1,1,0.5,0.5]
,[0.5,1,1,1.5,0.5,0.5,0.5,1]
,[1.5,1,0.5,0.5,1.5,0.5,0.5,2.5]
,[1,0.5,1,2,1.5,0.5,1,1]
,[1.5,0.5,2.5,1.5,2,1,1,0.5]
,[1,0.5,1,1.5,3.5,1,1,1.5]
,[1,0.5,1,0.5,0.5,0.5,0.5,0.5]])

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

#MapResults里task的编号从0开始，包含路由的task_graph的编号从1开始
def handle_my_output(tg_file_name,MapResults,task_graph,computation_ability,num_of_rows,output_file_name):
    hyperperiod,num_of_tasks,edges,comp_cost=init(tg_file_name)
    adj_matrix,total_needSend,total_needReceive,execution=Get_detailed_data(num_of_tasks,edges,comp_cost)
    ret_task_graph={}#任务编号从0开始，包括key和out_links
    for i in task_graph.keys():
        ret_task_graph.update({})
        task_graph[i].update({'input_links':[]})
        task_graph[i].update({'start_time':0})
        task_graph[i].update({'visited':0})
        task_graph[i].update({'total_needSend':total_needSend[int(i)]})
        task_graph[i].update({'end_time':0})
        task_graph[i].update({'total_needReceive':total_needReceive[int(i)]})
        mapto=MapResults[int(i)-1]
        task_graph[i].update({'mapto':mapto})
        row=int(mapto/num_of_rows)
        col=mapto%num_of_rows
        cur_computation_ability=computation_ability[row][col]
        task_graph[i].update({'exe_time':int(execution[int(i)]/cur_computation_ability)})
        #处理out_link
        for j in range(len(task_graph[i]['out_links'])):
            task_graph[i]['out_links'][j][0]=int(task_graph[i]['out_links'][j][0])
            task_graph[i]['out_links'][j].insert(2,[])
            task_graph[i]['out_links'][j][3]=[ task_graph[i]['out_links'][j][3] ]
            task_graph[i]['out_links'][j][-2]=mapto
            dest_position=MapResults[task_graph[i]['out_links'][j][0]-1]
            task_graph[i]['out_links'][j][-1]=dest_position
            task_graph[i]['out_links'][j]=[ task_graph[i]['out_links'][j] ]
            task_graph[i]['out_links'][j].append(0)
    
    #将task的编号改成从0开始，包括key和out_link里的task
    for i in task_graph.keys():
        cur_key=str(int(i)-1)
        ret_task_graph.update({cur_key:task_graph[i]})
        for j in range(len(ret_task_graph[cur_key]['out_links'])):
            ret_task_graph[cur_key]['out_links'][j][0][0]-=1
    
    with open(output_file_name,"w") as f:
        f.write(json.dumps(ret_task_graph,cls=MyEncoder))
    print("write done")

def handle_others_output(input_json,computation_ability,num_of_rows,output_file_name):
    task_graph={}
    with open(input_json,'r') as f1:
        task_graph=json.load(f1)

    for i in task_graph.keys():
        mapto=task_graph[i]['mapto']
        row=int(mapto/num_of_rows)
        col=mapto%num_of_rows
        cur_computation_ability=computation_ability[row][col]
        task_graph[i]['exe_time']=int(task_graph[i]['exe_time']/cur_computation_ability)
    
    with open(output_file_name,"w") as f2:
        f2.write(json.dumps(task_graph,cls=MyEncoder))
    print("write done")

#handle_my_output('./task graph/N12_autocor_AIR1.tgff',MapResults,task_graph,computation_ability,num_of_rows=8)

if __name__ == '__main__':
    json_list=['Autocor_Mesh8x8_AIR1__ra1.json','Autocor_Mesh8x8_AIR1__ra2.json','Autocor_Mesh8x8_AIR1_basic.json','Autocor_Mesh8x8_AIR1_improved.json','Autocor_Mesh8x8_AIR1_xy.json']
    for i in json_list:
        handle_others_output('./gem5_pending_results_10.12/'+i,computation_ability,num_of_rows=8,output_file_name='./gem5_pending_results_10.12/test_'+i)

