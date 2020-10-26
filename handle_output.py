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
            #task_graph[i]['out_links'][j].append(0)
    
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

    MapResults={0: 46, 1: 30, 2: 8, 3: 30, 4: 33, 5: 8, 6: 16, 7: 51, 8: 52, 9: 30, 10: 31, 11: 3, 12: 5, 13: 5, 14: 57, 15: 46, 16: 15, 17: 23, 18: 32, 19: 5, 20: 56, 21: 53, 22: 22, 23: 58, 24: 15, 25: 49, 26: 62, 27: 50, 28: 58, 29: 52, 30: 38}
    task_graph={'1': {'out_links': [['2', 28, [[46, 'N'], [38, 'N']], 0, 0, -1]]}, '2': {'out_links': [['3', 106, [[30, 'W'], [29, 'N'], [21, 'N'], [13, 'W'], [12, 'W'], [11, 'W'], [10, 'W'], [9, 'W']], 0, 0, -1]]}, '3': {'out_links': [['4', 119, [[8, 'E'], [9, 'S'], [17, 'S'], [25, 'E'], [26, 'E'], [27, 'E'], [28, 'E'], [29, 'E']], 0, 0, -1]]}, '4': {'out_links': [['6', 105, [[30, 'W'], [29, 'N'], [21, 'W'], [20, 'N'], [12, 'W'], [11, 'W'], [10, 'W'], [9, 'W']], 0, 0, -1], ['12', 74, [[30, 'N'], [22, 'W'], [21, 'N'], [13, 'N'], [5, 'W'], [4, 'W']], 0, 0, -1], ['18', 89, [[30, 'E'], [31, 'N']], 0, 0, -1], ['24', 49, [[30, 'S'], [38, 'S'], [46, 'S'], [54, 'W'], [53, 'W'], [52, 'S'], [60, 'W'], [59, 'W']], 0, 0, -1]]}, '6': {'out_links': [['8', 31, [[8, 'S'], [16, 'S'], [24, 'S'], [32, 'E'], [33, 'E'], [34, 'E'], [35, 'S'], [43, 'S']], 0, 0, -1], ['9', 112, [[8, 'E'], [9, 'E'], [10, 'S'], [18, 'S'], [26, 'S'], [34, 'S'], [42, 'S'], [50, 'E'], [51, 'E']], 0, 0, -1]]}, '12': {'out_links': [['14', 151, [[3, 'E'], [4, 'E']], 0, 0, -1], ['15', 126, [[3, 'S'], [11, 'S'], [19, 'S'], [27, 'W'], [26, 'W'], [25, 'S'], [33, 'S'], [41, 'S'], [49, 'S']], 0, 0, -1]]}, '18': {'out_links': [['20', 83, [[23, 'W'], [22, 'N'], [14, 'N'], [6, 'W']], 0, 0, -1], ['21', 133, [[23, 'S'], [31, 'W'], [30, 'W'], [29, 'S'], [37, 'W'], [36, 'S'], [44, 'W'], [43, 'W'], [42, 'W'], [41, 'W'], [40, 'S'], [48, 'S']], 0, 0, -1]]}, '24': {'out_links': [['26', 57, [[58, 'N'], [50, 'W']], 0, 0, -1], ['27', 153, [[58, 'E'], [59, 'E'], [60, 'E'], [61, 'E']], 0, 0, -1]]}, '26': {'out_links': [['25', 90, [[49, 'N'], [41, 'N'], [33, 'E'], [34, 'E'], [35, 'E'], [36, 'N'], [28, 'N'], [20, 'N'], [12, 'E'], [13, 'E'], [14, 'E']], 0, 0, -1]]}, '27': {'out_links': [['25', 101, [[62, 'E'], [63, 'N'], [55, 'N'], [47, 'N'], [39, 'N'], [31, 'N'], [23, 'N']], 0, 0, -1]]}, '14': {'out_links': [['13', 71, [], 0, 0, -1]]}, '15': {'out_links': [['13', 73, [[57, 'N'], [49, 'N'], [41, 'N'], [33, 'N'], [25, 'N'], [17, 'N'], [9, 'E'], [10, 'N'], [2, 'E'], [3, 'E'], [4, 'E']], 0, 0, -1]]}, '20': {'out_links': [['19', 105, [[5, 'S'], [13, 'W'], [12, 'W'], [11, 'S'], [19, 'W'], [18, 'S'], [26, 'S'], [34, 'W'], [33, 'W']], 0, 0, -1]]}, '21': {'out_links': [['19', 62, [[56, 'N'], [48, 'N'], [40, 'N']], 0, 0, -1]]}, '8': {'out_links': [['7', 60, [[51, 'W'], [50, 'N'], [42, 'W'], [41, 'W'], [40, 'N'], [32, 'N'], [24, 'N']], 0, 0, -1]]}, '9': {'out_links': [['7', 102, [[52, 'N'], [44, 'W'], [43, 'N'], [35, 'N'], [27, 'W'], [26, 'W'], [25, 'W'], [24, 'N']], 0, 0, -1]]}, '13': {'out_links': [['16', 79, [[5, 'E'], [6, 'S'], [14, 'S'], [22, 'S'], [30, 'S'], [38, 'S']], 0, 0, -1]]}, '25': {'out_links': [['28', 142, [[15, 'W'], [14, 'W'], [13, 'W'], [12, 'W'], [11, 'W'], [10, 'S'], [18, 'S'], [26, 'S'], [34, 'S'], [42, 'S']], 0, 0, -1]]}, '7': {'out_links': [['10', 30, [[16, 'S'], [24, 'E'], [25, 'E'], [26, 'E'], [27, 'E'], [28, 'E'], [29, 'E']], 0, 0, -1]]}, '19': {'out_links': [['22', 129, [[32, 'E'], [33, 'E'], [34, 'E'], [35, 'E'], [36, 'E'], [37, 'S'], [45, 'S']], 0, 0, -1]]}, '16': {'out_links': [['17', 50, [[46, 'N'], [38, 'N'], [30, 'N'], [22, 'E'], [23, 'N']], 0, 0, -1]]}, '22': {'out_links': [['23', 132, [[53, 'N'], [45, 'N'], [37, 'N'], [29, 'N'], [21, 'E']], 0, 0, -1]]}, '28': {'out_links': [['29', 96, [[50, 'S']], 0, 0, -1]]}, '10': {'out_links': [['11', 127, [[30, 'E']], 0, 0, -1]]}, '17': {'out_links': [['5', 83, [[15, 'S'], [23, 'S'], [31, 'S'], [39, 'W'], [38, 'W'], [37, 'W'], [36, 'W'], [35, 'W'], [34, 'W']], 0, 0, -1]]}, '11': {'out_links': [['5', 14, [[31, 'W'], [30, 'W'], [29, 'W'], [28, 'W'], [27, 'W'], [26, 'W'], [25, 'S']], 0, 0, -1]]}, '23': {'out_links': [['5', 87, [[22, 'W'], [21, 'W'], [20, 'W'], [19, 'W'], [18, 'S'], [26, 'S'], [34, 'W']], 0, 0, -1]]}, '5': {'out_links': [['30', 121, [[33, 'E'], [34, 'S'], [42, 'S'], [50, 'E'], [51, 'E']], 0, 0, -1]]}, '29': {'out_links': [['5', 34, [[58, 'W'], [57, 'N'], [49, 'N'], [41, 'N']], 0, 0, -1]]}, '30': {'out_links': [['31', 49, [[52, 'E'], [53, 'E'], [54, 'N'], [46, 'N']], 0, 0, -1]]}, '31': {'out_links': []}}

    computation_ability=read_NoC('./NoC description/N31_fmradio_Mesh8x8_NoCdescription.txt')

    dir_name='./gem5_pending_results_'+str(datetime.datetime.now().month)+'.'+str(datetime.datetime.now().day)+'/'

    if(os.path.exists(dir_name)==False):
        os.makedirs(dir_name)

    handle_my_output('./task graph/N31_fmradio.tgff',MapResults,task_graph,computation_ability,num_of_rows=8,output_file_name=dir_name+'fmradio_Mesh8x8_AIR1_co_optimization.json')
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


