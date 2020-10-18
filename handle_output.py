from libs import init,Get_detailed_data
import numpy as np
import json
import os
import datetime


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
        task_graph[i].update({'exe_time':computation_ability[int(i)-1][mapto]})
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
        task_graph[i]['exe_time']=computation_ability[int(i)][mapto]
    
    with open(output_file_name,"w") as f2:
        f2.write(json.dumps(task_graph,cls=MyEncoder))
    print("write done")


def read_NoC(NoC_file_name):
    ret=[]
    f=open(NoC_file_name)
    for line in f:
        tmp=[]
        for i in line[1:-2].split(','):
            tmp.append(int(i))
        ret.append(tmp)
    return ret


if __name__ == '__main__':
    dir_name='./gem5_pending_results_'+str(datetime.datetime.now().month)+'.'+str(datetime.datetime.now().day)+'/'

    if(os.path.exists(dir_name)==False):
        os.makedirs(dir_name)

    MapResults={0: 37, 1: 28, 2: 3, 3: 18, 4: 10, 5: 54, 6: 38, 7: 18, 8: 17, 9: 24, 10: 23, 11: 3, 12: 8, 13: 18, 14: 6, 15: 1, 16: 39, 17: 6, 18: 52, 19: 51, 20: 57, 21: 9}
    task_graph={'1': {'out_links': [['2', 74, [[37, 'W'], [36, 'N']], 0, 0, -1]]}, '2': {'out_links': [['4', 24, [[28, 'W'], [27, 'N'], [19, 'W']], 0, 0, -1]]}, '4': {'out_links': [['6', 37, [[18, 'E'], [19, 'E'], [20, 'S'], [28, 'S'], [36, 'E'], [37, 'E'], [38, 'S'], [46, 'S']], 0, 0, -1], ['7', 92, [[18, 'S'], [26, 'S'], [34, 'E'], [35, 'E'], [36, 'E'], [37, 'E']], 0, 0, -1], ['8', 97, [], 0, 0, -1], ['9', 15, [[18, 'W']], 0, 0, -1], ['10', 64, [[18, 'W'], [17, 'W'], [16, 'S']], 0, 0, -1], ['11', 12, [[18, 'E'], [19, 'E'], [20, 'E'], [21, 'E'], [22, 'E']], 0, 0, -1], ['12', 33, [[18, 'N'], [10, 'E'], [11, 'N']], 0, 0, -1], ['13', 12, [[18, 'N'], [10, 'W'], [9, 'W']], 0, 0, -1], ['14', 22, [], 0, 0, -1], ['15', 11, [[18, 'N'], [10, 'N'], [2, 'E'], [3, 'E'], [4, 'E'], [5, 'E']], 0, 0, -1], ['16', 76, [[18, 'W'], [17, 'N'], [9, 'N']], 0, 0, -1], ['17', 14, [[18, 'S'], [26, 'E'], [27, 'E'], [28, 'E'], [29, 'E'], [30, 'E'], [31, 'S']], 0, 0, -1], ['18', 11, [[18, 'N'], [10, 'N'], [2, 'E'], [3, 'E'], [4, 'E'], [5, 'E']], 0, 0, -1], ['19', 62, [[18, 'E'], [19, 'E'], [20, 'S'], [28, 'S'], [36, 'S'], [44, 'S']], 0, 0, -1], ['20', 15, [[18, 'S'], [26, 'E'], [27, 'S'], [35, 'S'], [43, 'S']], 0, 0, -1]]}, '6': {'out_links': [['5', 1, [[54, 'W'], [53, 'W'], [52, 'W'], [51, 'N'], [43, 'N'], [35, 'N'], [27, 'N'], [19, 'N'], [11, 'W']], 0, 0, -1]]}, '7': {'out_links': [['5', 10, [[38, 'N'], [30, 'N'], [22, 'W'], [21, 'W'], [20, 'W'], [19, 'N'], [11, 'W']], 0, 0, -1]]}, '8': {'out_links': [['5', 11, [[18, 'N']], 0, 0, -1]]}, '9': {'out_links': [['5', 12, [[17, 'N'], [9, 'E']], 0, 0, -1]]}, '10': {'out_links': [['5', 12, [[24, 'N'], [16, 'N'], [8, 'E'], [9, 'E']], 0, 0, -1]]}, '11': {'out_links': [['5', 56, [[23, 'N'], [15, 'W'], [14, 'W'], [13, 'W'], [12, 'W'], [11, 'W']], 0, 0, -1]]}, '12': {'out_links': [['5', 37, [[3, 'W'], [2, 'S']], 0, 0, -1]]}, '13': {'out_links': [['5', 11, [[8, 'E'], [9, 'E']], 0, 0, -1]]}, '14': {'out_links': [['5', 61, [[18, 'N']], 0, 0, -1]]}, '15': {'out_links': [['5', 12, [[6, 'W'], [5, 'W'], [4, 'W'], [3, 'W'], [2, 'S']], 0, 0, -1]]}, '16': {'out_links': [['5', 26, [[1, 'E'], [2, 'S']], 0, 0, -1]]}, '17': {'out_links': [['5', 37, [[39, 'W'], [38, 'N'], [30, 'N'], [22, 'N'], [14, 'W'], [13, 'W'], [12, 'W'], [11, 'W']], 0, 0, -1]]}, '18': {'out_links': [['5', 21, [[6, 'W'], [5, 'W'], [4, 'W'], [3, 'W'], [2, 'S']], 0, 0, -1]]}, '19': {'out_links': [['5', 20, [[52, 'W'], [51, 'N'], [43, 'N'], [35, 'N'], [27, 'N'], [19, 'W'], [18, 'N']], 0, 0, -1]]}, '20': {'out_links': [['5', 12, [[51, 'N'], [43, 'N'], [35, 'N'], [27, 'N'], [19, 'N'], [11, 'W']], 0, 0, -1]]}, '5': {'out_links': [['21', 13, [[10, 'S'], [18, 'S'], [26, 'S'], [34, 'W'], [33, 'S'], [41, 'S'], [49, 'S']], 0, 0, -1]]}, '21': {'out_links': [['3', 82, [[57, 'E'], [58, 'E'], [59, 'N'], [51, 'N'], [43, 'N'], [35, 'N'], [27, 'N'], [19, 'N'], [11, 'N']], 0, 0, -1]]}, '3': {'out_links': [['22', 85, [[3, 'W'], [2, 'S'], [10, 'W']], 0, 0, -1]]}, '22': {'out_links': []}}

    computation_ability=read_NoC('./NoC description/N22_audiobeam_Mesh8x8_NoCdescription.txt')


    handle_my_output('./task graph/N22_audiobeam.tgff',MapResults,task_graph,computation_ability,num_of_rows=8,output_file_name=dir_name+'audiobeam_Mesh8x8_AIR1_co_optimization.json')
    """
    json_list=[]
    for i in os.listdir('./'):
        if(i[-5:]=='.json'):
            json_list.append(i)
    print("handle json files:")
    print(json_list)
    for i in json_list:
        handle_others_output(i,computation_ability,num_of_rows=8,output_file_name=dir_name+'_test_'+i)
    """


