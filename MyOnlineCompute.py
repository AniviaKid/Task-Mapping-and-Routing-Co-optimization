import sys
import getopt
import json
import math
#import Queue
import networkx as nx
import pylab
import numpy as np
import logging, sys
import copy
from libs import Get_link_connection_by_index,Check_contention
import queue



class linklist():
    def __init__(self):
        '''
        初始化节点
        
        |type reserveList:
        |
        '''
        self.nList = 0
        self.sList = 0
        self.wList = 0
        self.eList = 0
        self.pList = 0


class link_item():#可以根据在list中的下标索引到它连接的是哪两个PE
    def __init__(self):
        #记录这个link的timeline，list中的每个元素是list，形式为[task_source,task_dest,start_time,end_time]
        self.timeline=[]
        

class onlineTimeline: 
    def __init__(self,inputfile,rowNum):
        self.inputfile =inputfile
        self.totalNum = int(rowNum)*int(rowNum)
        self.rowNum = int(rowNum)
        self.NoClink = []
     

        for i in range(0,self.totalNum):
            link = linklist()
            
            self.NoClink.append(link)
         

        self.sendMatrix = [-1]
        self.receiveMatrix = [-1]
        self.totalSize=0
        self.exeMatric = [-1]
        self.taskGraph = {}
        self.pendTask = []
        self.routeNow = {}
       
        self.stateMatrix=[3]
        self.sendingNow = []
        self.MapResult=[]

        self.nowPri = -1

    

        self.nowTime = 0
        self.num_of_tasks=0

        self.fullRouteFromRL=[]#(task_source,task_dest)表示从task_source到task_dest的route全部是由RL计算的
        self.partRouteFromRL=[]#前两个元素i，j表示在计算从task_i到task_j的出边，之后的元素就是json里的route格式，如[0, "S"]，当前位置+下一步移动方向
        self.pendTimes=0#由RL计算的路径导致推迟的次数

        #仿真器2需要用的东西
        self.adj_matrix=[[]]#邻接矩阵，task下标从1开始
        self.edge_set={}#'2,3':{'transmission':10,'used_link':[]}
        self.link_set=[]
        total_link_num=(rowNum-1+rowNum)*(rowNum-1)+rowNum-1
        for i in range(0,total_link_num):
            tmp=link_item()
            self.link_set.append(tmp)
        self.task_end_time=[-1]#-1表示这个位置的task不存在，0表示它的end_time还没有确定
        self.partRouteFromRL_link_index=[]
        


    #这里load的时候task_graph里的下标是从1开始的，key是str格式的
    def loadGraphByDict(self,taskGraph1,MapResult1,fullRouteFromRL1,partRouteFromRL1,num_of_tasks):
        tmp_taskgraph=copy.deepcopy(taskGraph1)#task graph里的task下标是不连续的
        self.num_of_tasks=num_of_tasks
        #print(tmp_taskgraph)
        for i in range(1,num_of_tasks+1):
            if(str(i) in tmp_taskgraph.keys()):
                self.sendMatrix.append(tmp_taskgraph[str(i)]['total_needSend'])
                self.receiveMatrix.append(tmp_taskgraph[str(i)]['total_needReceive'])
                self.totalSize = self.totalSize + tmp_taskgraph[str(i)]['total_needSend']
                self.exeMatric.append(tmp_taskgraph[str(i)]['exe_time'])
                self.stateMatrix.append(1000)
                self.task_end_time.append(0)
                for task in tmp_taskgraph[str(i)]['out_links']:  
                    task.append(0)
            else:
                #这里我为了写自己的仿真器改了，不然的话应该是append 0 0 0 3
                self.sendMatrix.append(-1)
                self.receiveMatrix.append(-1)
                self.exeMatric.append(-1)
                self.stateMatrix.append(3)
                self.task_end_time.append(-1)
        self.taskGraph = tmp_taskgraph
        self.MapResult = MapResult1
        self.fullRouteFromRL=fullRouteFromRL1
        self.partRouteFromRL=partRouteFromRL1
        self.stateMatrix[1]=1
        
        print("sendMatric",self.sendMatrix)
        print("receiveMatric",self.receiveMatrix)
        print("exeMatric",self.exeMatric)
        print("taskGraph",self.taskGraph)
        print("MapResult",self.MapResult)
        print("stateMatric,",self.stateMatrix)
        print("fullRouteFromRL",self.fullRouteFromRL)
        print("partRouteFromRL",self.partRouteFromRL)
        
        self.adj_matrix=np.zeros((num_of_tasks+1,num_of_tasks+1),dtype=np.int)
        for i in self.taskGraph.keys():
            for j in self.taskGraph[i]['out_links']:#j=["2", 2, [], 0, 0, -1]
                self.adj_matrix[int(i)][int(j[0])]=j[1]
                tmp_key=str(i)+','+str(j[0])
                used_link=[]
                for k in j[2]:
                    tmp_row=int(k[0]/self.rowNum)
                    tmp_col=k[0]%self.rowNum
                    if(k[1]=='N'):
                        tmp_row-=1
                        used_link.append( (2*self.rowNum-1)*tmp_row+(self.rowNum-1)+tmp_col )
                    elif(k[1]=='S'):
                        used_link.append( (2*self.rowNum-1)*tmp_row+(self.rowNum-1)+tmp_col )
                    elif(k[1]=='W'):
                        used_link.append( (2*self.rowNum-1)*tmp_row+(tmp_col-1) )
                    elif(k[1]=='E'):
                        used_link.append( (2*self.rowNum-1)*tmp_row+tmp_col )
                self.edge_set.update({tmp_key:{'transmission':j[1],'used_link':used_link}})
        self.partRouteFromRL_link_index.append(self.partRouteFromRL[0])
        self.partRouteFromRL_link_index.append(self.partRouteFromRL[1])
        for i in range(2,len(self.partRouteFromRL)):
            tmp_row=int(self.partRouteFromRL[i][0]/self.rowNum)
            tmp_col=self.partRouteFromRL[i][0]%self.rowNum
            if(self.partRouteFromRL[i][1]=='N'):
                tmp_row-=1
                self.partRouteFromRL_link_index.append( (2*self.rowNum-1)*tmp_row+(self.rowNum-1)+tmp_col )
            elif(self.partRouteFromRL[i][1]=='S'):
                self.partRouteFromRL_link_index.append( (2*self.rowNum-1)*tmp_row+(self.rowNum-1)+tmp_col )
            elif(self.partRouteFromRL[i][1]=='W'):
                self.partRouteFromRL_link_index.append( (2*self.rowNum-1)*tmp_row+(tmp_col-1) )
            elif(self.partRouteFromRL[i][1]=='E'):
                self.partRouteFromRL_link_index.append( (2*self.rowNum-1)*tmp_row+tmp_col )
        #print(self.adj_matrix)
        #print(self.edge_set)
        

        
        
    def loadGraph(self):
        with open(self.inputfile+"taskGraph.json","r") as f:
            taskGraph1 = json.load(f)
            sumR= 0;
            sumS=0
           

            for i in range(1,len(taskGraph1)+1):
                self.sendMatrix.append(taskGraph1[str(i)]['total_needSend'])
                self.receiveMatrix.append(taskGraph1[str(i)]['total_needReceive'])
                self.totalSize = self.totalSize + taskGraph1[str(i)]['total_needSend']
                self.exeMatric.append(taskGraph1[str(i)]['exe_time'])
                
                self.stateMatrix.append(1000)
                for task in taskGraph1[str(i)]['out_links']:
                    
                    task.append(0)
                    
                    #self.router[i].startedSending[task[0]] = 0
                    #print(i,task,self.router[i].startedSending[task[0]] )

            self.taskGraph = taskGraph1
        with open(self.inputfile+"MapResult.json","r") as f:
            self.MapResult = json.load(f)

        print("task graph loaded++++++++++++++++++++++")
        print("sendMatric",self.sendMatrix)
        print("receiveMatric",self.receiveMatrix)
        print("exeMatric",self.exeMatric)
        
        self.stateMatrix[1]=1
        
        
       
        print("taskGraph",self.taskGraph)
        print("MapResult",self.MapResult)
   
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")



    def findStartExe(self):
        exeThisTime=[]
        for i in range(1,self.num_of_tasks+1):
            if(self.receiveMatrix[i]==0 and self.exeMatric[i]!=0):
                exeThisTime.append(i)
        return exeThisTime


    def startExe(self,exeThisTime):
        for i in exeThisTime:
            self.exeMatric[i]=self.exeMatric[i]-1


    def findMapZone(self,father):
        mapZone = []
        params = [-2,-1,+1,+2]
        params1 = [-self.rowNum*2,-self.rowNum,self.rowNum*2,self.rowNum]
        params2 = [-self.rowNum-1,-self.rowNum+1]
        params3 = [self.rowNum-1,self.rowNum+1]
        for param in params1:
            t = father+param
            if(t<64 and t>=0):
                mapZone.append(t)

        for param in params:
            t = father+param
            z = int(father/8)
            if(t<(z*8+8) and t>=z*8):
                mapZone.append(t)

        for param in params2:
            t = father+param
            z = int(father/8)-1
            if(t<(z*8+8) and t>=z*8):
                mapZone.append(t)

        for param in params3:
            t = father+param
            z = int(father/8)+1
            if(t<(z*8+8) and t>=z*8):
                mapZone.append(t)
        return mapZone


    def findStartSend(self):
        sendThisTime=[]
        for i in range(1,self.num_of_tasks+1):
            if(self.sendMatrix[i]!=0 and self.exeMatric[i]==0 and self.stateMatrix[i]!=3):
                sendThisTime.append(i)
        return sendThisTime


  
    def sending(self,sendinfos):
        releaseList = []
        for item in sendinfos:
            
            for link in self.taskGraph[str(item[0])]['out_links'][:]:#检索正在sending的出边，它的source的所有出边
                if(link[0]==str(item[1])):#如果检索到的这条出边跟正在sending的出边是同一条边
                    
                    # Sending Complete!

                    if(link[1]==0):#要send的size=0，就是这条出边已经send完了
                        self.taskGraph[str(item[0])]['out_links'].remove(link)
                        releaseList.append(item)
                        # Update Priority
                        if(self.nowPri<link[-2]):#如果nowPri比正在send的这条出边的优先级低，更新nowPri
                            self.nowPri = link[-2]
                        break
                    link[1]=link[1]-1#每次send 1个size
                    self.sendMatrix[item[0]]=self.sendMatrix[item[0]]-1
                    self.receiveMatrix[item[1]]=self.receiveMatrix[item[1]]-1
                    break
            
                
        return releaseList

    def reserveRoute(self,route):
        for rt in route:
            if(rt[1] == 'E'):
                self.NoClink[rt[0]].eList = 1
            elif(rt[1] == 'W'):
                self.NoClink[rt[0]].wList = 1
            elif(rt[1] == 'N'):
                self.NoClink[rt[0]].nList = 1
            elif(rt[1] == 'S'):
                self.NoClink[rt[0]].sList = 1


    def printNoC(self):
        for i in range(0,len(self.NoClink)):
            print(i,self.NoClink[i].nList,self.NoClink[i].sList,self.NoClink[i].wList,self.NoClink[i].eList,end=" | ")
     
     #Task State: Not start yet 1000
     #source sending 1
     #source Pending 2
     #source finished 3  
     #dest  sending -1
     #dest Pending -2
     #dest Finished -3 

    def sendPackage(self,sendThisTime):
        releaseList = []
        
        for i in sendThisTime:
            #print(self.taskGraph[str(i)]['out_links'])
            for dest in self.taskGraph[str(i)]['out_links'][:]:#检查该task的所有出边，初始化的时候每条出边末尾默认添加了一个0
                # Not start yet
                if(dest[-1]==0):
                    sendtoi = int(dest[0])
                    #if send or receive already finish
                    #print(sendtoi)
                    route = dest[2]

                    if(len(route)==0):
                        self.sendMatrix[i]=self.sendMatrix[i]-dest[1]
                        self.receiveMatrix[sendtoi]=self.receiveMatrix[sendtoi]-dest[1]
                        dest[-1]=3
                        canSend=-1
                        #print("Since Mapped to same core, this transmiss = 0", i,sendtoi)
                        self.taskGraph[str(i)]['out_links'].remove(dest)
                    else:  
                        canSend = self.checkCanSend(route,dest[-2],dest[1],int(i),int(dest[0]))#route,priority,size,task_source,task_destination
                    if(canSend==0):
                        #print("should Pending",sendtoi)
                        if([i,sendtoi,route,dest] not in self.pendTask):
                            
                            self.stateMatrix[sendtoi]=-2
                            self.stateMatrix[i]=2
                            self.pendTask.append([i,sendtoi,route,dest])
                            #self.taskGraph[str(i)]['out_links'].remove(dest)
                            dest[-1]=2
                    elif(canSend==1):
                        #Start to transmiss
                        #print("checking can send=",sendtoi)

                        #self.stateMatrix[sendtoi]=-1
                        #self.stateMatrix[i]=1
                        if([i,sendtoi,route,dest] not in self.sendingNow):
                            self.sendingNow.append([i,sendtoi,route,dest])
                            #self.taskGraph[str(i)]['out_links'].remove(dest)
                            dest[-1]=1
                            self.reserveRoute(route)


        for i in self.pendTask[:]:
            canSend = self.checkCanSend(i[2],i[-1][-2],i[-1][1],int(i[0]),int(i[1]))

            if(canSend==1):
                #print("from pending",i)
                self.stateMatrix[i[1]]=-1
                
                #self.stateMatrix[i[0]]=1

                self.reserveRoute(i[2])
                self.sendingNow.append(i)
                #releaseList.append(self.sending(i[0],i[1],i[2]))
                self.pendTask.remove(i)


        reList=[]
        reList=self.sending(self.sendingNow)#每个item是[i,sendtoi,route,dest]
        if(reList!=None and len(reList)!=0):
            releaseList=reList

        return releaseList

    #if stateMatrix==3 deleted
    def releaseRec(self,releaseList):
        #print("releaseNow,",releaseList)
        for item in releaseList:#item：source，dest，route，整个这个out_link的信息（des,size,route,starttime,endtime,pri）
            if(item!=None):

                if(self.receiveMatrix[item[1]]==0):
                    self.stateMatrix[item[1]]=-3
                if(self.sendMatrix[item[0]]==0):
                    self.stateMatrix[item[0]]=3


                for rt in item[2]:
                    if(rt[1] == 'E'):
                        self.NoClink[rt[0]].eList = 0
                    elif(rt[1] == 'W'):
                        self.NoClink[rt[0]].wList = 0
                    elif(rt[1] == 'N'):
                        self.NoClink[rt[0]].nList = 0
                    elif(rt[1] == 'S'):
                        self.NoClink[rt[0]].sList = 0


                
                if(len(self.taskGraph[str(item[0])]['out_links'])==0):
                    self.stateMatrix[item[0]]=3

                self.sendingNow.remove(item)
                #print(self.sendingNow)
        #print("after releasing")
        #self.printNoC()


    def computeTime(self):
        z= 0
        remain = 1000
        releaseList = []
        while(remain>0):
            
            

            #print("In the cycle +++++++++++++++++----------------------------")
            #print(self.nowTime)
            signal = 0
            remain = 0

            

            sendThisTime =self.findStartSend()
            #print("sendThisTime",sendThisTime)
            exeThisTime = self.findStartExe()
            #print("exeThistime",exeThisTime)

            releaseList = self.sendPackage(sendThisTime)
            

            

            #Find can exe this time 
            

            self.startExe(exeThisTime)


            #self.checkSend()


            self.releaseRec(releaseList)
            """
            ###print Network State
            self.Dprint("sendMatric",self.sendMatrix)
            self.Dprint("receiveMatric",self.receiveMatrix)
            self.Dprint("exeMatric",self.exeMatric)
            self.Dprint1("sendingNow",self.sendingNow)
            self.Dprint1("releaseList",releaseList)
            self.Dprint1("reserveList",self.pendTask)
            self.Dprint("stateMatrix",self.stateMatrix)
            print("Priority now is ",self.nowPri)
            #print("-----------------------------")
            #print("task 3",self.taskGraph['3'])
            self.printNoC()
            """

            for i in range(0,len(self.receiveMatrix)):
                remain=remain+self.receiveMatrix[i]+self.exeMatric[i]


            self.nowTime=self.nowTime+1
            # if(self.nowTime>0):
            #     str1 = input()
    
        #print("total time:",self.nowTime)
        return self.pendTimes

    def Dprint(self,name1,list1):
        print(name1)
        i=0
        for i in range(0,len(list1)):
            print(i,list1[i],end=' | ')
        print('\n')

    def Dprint1(self,name1,list1):
        print(name1)
        i=0
        for i in range(0,len(list1)):
            print(list1[i])
        

    
    def computeRoute(self,i,dst):
        route = []
        (srcX,srcY) = self.changeIndex(i)
        (dstX,dstY) = self.changeIndex(dst)
        
        while(dstY > srcY):
            route.append([i,'E'])
            srcY = srcY + 1
            i = i + 1

        while(dstY < srcY):
            route.append([i,'W'])
            srcY = srcY - 1
            i = i - 1

        while(dstX > srcX):
            route.append([i,'S'])
            srcX = srcX +1
            i = i + self.rowNum

        while(dstX < srcX):
            route.append([i,'N'])
            srcX = srcX - 1
            i = i - self.rowNum

        return route        
            

    def checkCanSend(self,route,pri,size,task_source,task_destination):#传进来的task都是int类型
        #print("this is for checking",route,pri,size,task_source,task_destination)
        #print("checking resout+++++++++++++")
        flag=False
        for i in self.fullRouteFromRL:
            if(task_source == i[0] and task_destination == i[1]):#这条出边全部由RL计算
                flag=True
                break
        for rt in route:
            #print(rt,self.NoClink[rt[0]].eList,self.NoClink[rt[0]].wList,self.NoClink[rt[0]].nList,self.NoClink[rt[0]].sList)
            if(rt[1] == 'E' and self.NoClink[rt[0]].eList == 1):
                if(flag):
                    self.pendTimes+=1
                elif(task_source==self.partRouteFromRL[0] and task_destination==self.partRouteFromRL[1] and rt in self.partRouteFromRL):
                    self.pendTimes+=1
                return False
            elif(rt[1] == 'W' and self.NoClink[rt[0]].wList == 1):
                if(flag):
                    self.pendTimes+=1
                elif(task_source==self.partRouteFromRL[0] and task_destination==self.partRouteFromRL[1] and rt in self.partRouteFromRL):
                    self.pendTimes+=1
                return False
            elif(rt[1] == 'N' and self.NoClink[rt[0]].nList == 1):
                if(flag):
                    self.pendTimes+=1
                elif(task_source==self.partRouteFromRL[0] and task_destination==self.partRouteFromRL[1] and rt in self.partRouteFromRL):
                    self.pendTimes+=1
                return False
            elif(rt[1] == 'S' and self.NoClink[rt[0]].sList == 1):
                if(flag):
                    self.pendTimes+=1
                elif(task_source==self.partRouteFromRL[0] and task_destination==self.partRouteFromRL[1] and rt in self.partRouteFromRL):
                    self.pendTimes+=1
                return False
        '''
        for i in range(1,len(self.taskGraph)+1):#检查每一个task
            for edge in self.taskGraph[str(i)]['out_links']:#检查当前task的每一个出边
                if(edge[-2]<pri):#如果存在某一条边的优先级比它小
                    for link in edge[2]:#检查这条出边的所有route
                        for rt in route:
                            #if there is overlap between route
                            if(link==rt):
                                if((edge[-1]==0) and (self.nowTime+size)>=edge[3]):#现在的时间+size>=这条优先级较小的边的开始时间
                                    return False
        '''
        return True

    def changeName(self,i,dst):
        return str(i)+'+'+str(dst)

    def startSending(self,i,dst,route):
        self.routeNow[self.changeName(i,dst)] = route
        print("this is starting------------------------",i,dst)
        
        for rt in route:
            if(rt[1] == 'E'):
                self.NoClink[rt[0]].eList = 1
            elif(rt[1] == 'W'):
                self.NoClink[rt[0]].wList = 1
            elif(rt[1] == 'N'):
                self.NoClink[rt[0]].nList = 1
            elif(rt[1] == 'S'):
                self.NoClink[rt[0]].sList = 1

        #for i in range(0,50):
           #print(rt,self.NoClink[i].eList,self.NoClink[i].wList,self.NoClink[i].nList,self.NoClink[i].sList)

    def changeIndex(self,index):
        return (int(int(index)/self.rowNum),int(int(index)%self.rowNum))

    def computeContention(self):
        contention=0
        edge_queue=[]#每个item为( 'task_source,task_dest' , end time of task_source )，如('1,2',20)
        #添加一开始就能执行的边
        for i in range(1,self.num_of_tasks+1):
            if(self.receiveMatrix[i] == 0):#这个task可以立刻执行，然后开始传输
                for j in self.taskGraph[str(i)]['out_links']:
                    tmp=(str(i)+','+j[0],self.exeMatric[i])
                    edge_queue.append(tmp)
                self.task_end_time[i]=self.exeMatric[i]
        edge_queue.sort(key=lambda x: x[1])#按照task_source的结束时间排序

        #队列不空时，取队首的边来执行，需要检查这条边占用的所有link在[task_source的结束时间,task_source的结束时间+transmission]时间段是否可用，如果可用的话就占用这个时间段的这些link，如果不可用，则等待时间T，直到[task_source的结束时间+T,task_source的结束时间+T+transmission]，而这个时间T就是这条边的contention
        while(len(edge_queue)!=0):
            current_edge=edge_queue[0]#( 'task_source,task_dest' , end time of task_source )
            edge_queue.pop(0)
            contended_link=[]
            contended_timeInterval_index=[]#这里留的是对应link上发生争用的区间的索引
            current_end_time_of_source=current_edge[1]
            current_end_time_of_dest=0
            current_transmission=self.edge_set[current_edge[0]]['transmission']
            for i in self.edge_set[current_edge[0]]['used_link']:#检查这条边用过的link
                for j in self.link_set[i].timeline:#检查这条link的时间轴
                    if(current_end_time_of_source+current_transmission>j[2] and current_end_time_of_source+current_transmission<j[3]):#目标区间右侧落在了当前检测到的区间里，冲突
                        contended_link.append(i)
                        contended_timeInterval_index.append(self.link_set[i].timeline.index(j))
                    elif(current_end_time_of_source>j[2] and current_end_time_of_source<j[3]):#目标区间左侧落在了当前检测到的区间里，冲突
                        contended_link.append(i)
                        contended_timeInterval_index.append(self.link_set[i].timeline.index(j))
                    elif(current_end_time_of_source<j[2] and current_end_time_of_source+current_transmission>j[3]):#目标区间包括了当前检测到的区间，冲突
                        contended_link.append(i)
                        contended_timeInterval_index.append(self.link_set[i].timeline.index(j))
            if(len(contended_link)==0):#这次要传输的edge没有任何争用，那么直接将传输任务更新到link的时间轴上
                for i in self.edge_set[current_edge[0]]['used_link']:#遍历这条边的所有link
                    if(len(self.link_set[i].timeline)==0):#这个link还没有被使用过，直接添加
                        self.link_set[i].timeline.append([int(current_edge[0][0]),int(current_edge[0][2]),current_end_time_of_source,current_end_time_of_source+current_transmission])

                    for j in range(len(self.link_set[i].timeline)):#访问这条link的时间轴
                        if(current_end_time_of_source+current_transmission<=self.link_set[i].timeline[j][2]):#放在这个区间的左边
                            self.link_set[i].timeline.insert(j,
                            [int(current_edge[0][0]),int(current_edge[0][2]),current_end_time_of_source,current_end_time_of_source+current_transmission]
                            )
                            break
                        elif(current_end_time_of_source>=self.link_set[i].timeline[j][3]):#放在这个区间的右边
                            if(j==(len(self.link_set[i].timeline)-1)):#最后一位的右边，用append
                                self.link_set[i].timeline.append([int(current_edge[0][0]),int(current_edge[0][2]),current_end_time_of_source,current_end_time_of_source+current_transmission])
                            else:
                                self.link_set[i].timeline.insert(j+1,
                                [int(current_edge[0][0]),int(current_edge[0][2]),current_end_time_of_source,current_end_time_of_source+current_transmission]
                                )
                            break
                current_end_time_of_dest=current_end_time_of_source+current_transmission
            else:#这次要传输的边有争用，需要计算参数T
                T_=0
                is_from_RL=False#当前访问的这个link是否来自RL的计算
                for i in self.fullRouteFromRL:
                    if(i[0]==int(current_edge[0][0]) and i[1]==int(current_edge[0][2])):
                        is_from_RL=True
                if(is_from_RL==False):#这条边不是全部由RL计算的，需要判断是否有部分是RL计算的
                    tmp_partRoute=copy.deepcopy(self.partRouteFromRL_link_index)
                    tmp_partRoute.pop(0)
                    tmp_partRoute.pop(0)
                    for i in contended_link:#检查发生过争用的link是否来自RL
                        if(int(current_edge[0][0])==self.partRouteFromRL_link_index[0] and int(current_edge[0][2])==self.partRouteFromRL_link_index[1] and i in tmp_partRoute):#这条link是由RL计算的
                            is_from_RL=True
                            break
                max_end_time=0
                for i in range(0,len(contended_timeInterval_index)):#寻找争用过的link的最大结束时间
                    if(self.link_set[ contended_link[i] ].timeline[ contended_timeInterval_index[i][3] ] > max_end_time):
                        max_end_time=self.link_set[ contended_link[i] ].timeline[ contended_timeInterval_index[i][3] ]
                T_=max_end_time-current_end_time_of_source#设置T值的初始值，然后还需要检测这个T是否可以使得这条边传输的时候不争用
                flag_contention,link_index,timeInterval_index=Check_contention(self.edge_set[current_edge[0]]['used_link'],self.link_set,current_end_time_of_source+T_,current_end_time_of_source+T_+current_transmission)
                while(flag_contention==False):#增加T后依然发生争用，还需要继续增大T
                    T_=self.link_set[link_index].timeline[timeInterval_index][3]-current_end_time_of_source
                    flag_contention,link_index,timeInterval_index=Check_contention(self.edge_set[current_edge[0]]['used_link'],self.link_set,current_end_time_of_source+T_,current_end_time_of_source+T_+current_transmission)
                if(is_from_RL==True):
                    contention+=T_
                #更新link_set中的时间轴
                for i in self.edge_set[current_edge[0]]['used_link']:#遍历这条边的所有link
                    for j in range(len(self.link_set[i].timeline)):#访问这条link的时间轴
                        if(current_end_time_of_source+T_+current_transmission<=self.link_set[i].timeline[j][2]):#放在这个区间的左边
                            self.link_set[i].timeline.insert(j,
                            [int(current_edge[0][0]),int(current_edge[0][2]),current_end_time_of_source,current_end_time_of_source+current_transmission]
                            )
                            break
                        elif(current_end_time_of_source+T_>=self.link_set[i].timeline[j][3]):#放在这个区间的右边
                            if(j==(len(self.link_set[i].timeline)-1)):#最后一位的右边，用append
                                self.link_set[i].timeline.append([int(current_edge[0][0]),int(current_edge[0][2]),current_end_time_of_source,current_end_time_of_source+current_transmission])
                            else:
                                self.link_set[i].timeline.insert(j+1,
                                [int(current_edge[0][0]),int(current_edge[0][2]),current_end_time_of_source,current_end_time_of_source+current_transmission]
                                )
                            break
                current_end_time_of_dest=current_end_time_of_source+T_+current_transmission
            #传输结束后，需要计算是否有新的边可以加入队列
            self.receiveMatrix[int(current_edge[0][2])]-=self.edge_set[current_edge]['transmission']
            if(self.receiveMatrix[int(current_edge[0][2])]==0):#task_dest已经可以执行，那么将它的出边加入到队列中
                for i in self.taskGraph[current_edge[0][2]]['out_links']:
                    edge_queue.append( (current_edge[0][2]+','+i[0] , current_end_time_of_dest) )
                edge_queue.sort(key=lambda x: x[1])#按照task_source的结束时间排序
        return contention
                


                
                    

def main(argv):
    inputfile = ''
    rowNum = ''
    """
    try:
        opts, args = getopt.getopt(argv,"hi:r:",["ifile=","row="])
    except getopt.GetoptError:
        print('Error Online.py -i <inputfile> -r <row>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Online.py -i <inputfile> -r <row>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            print("read input file")
            inputfile = arg
        elif opt in ("-r", "--row"):
            rowNum = arg
    """
    MapResult=[0,29,34,28,3,22,17,13,42,55,8,60,6]
    TaskGraph11={'1': {'total_needSend': 190, 'out_links': [['2', 190, [[29, 'W'], [28, 'W'], [27, 'W'], [26, 'S']], 0, 0, -1]], 'total_needReceive': 0, 'exe_time': 88}, '2': {'total_needSend': 750, 'out_links': [['8', 60, [[34, 'S']], 0, 0, -1], ['4', 70, [[34, 'N'], [26, 'N'], [18, 'N'], [10, 'N'], [2, 'E']], 0, 0, -1], ['7', 70, [[34, 'E'], [35, 'N'], [27, 'N'], [19, 'E'], [20, 'E'], [21, 'N']], 0, 0, -1], ['10', 80, [[34, 'W'], [33, 'N'], [25, 'N'], [17, 'W'], [16, 'N']], 0, 0, -1], ['9', 100, [[34, 'S'], [42, 'S'], [50, 'E'], [51, 'E'], [52, 'E'], [53, 'E'], [54, 'E']], 0, 0, -1], ['11', 100, [[34, 'E'], [35, 'E'], [36, 'S'], [44, 'S'], [52, 'S']], 0, 0, -1], ['6', 130, [[34, 'W'], [33, 'N'], [25, 'N']], 0, 0, -1], ['5', 140, [[34, 'N'], [26, 'N'], [18, 'E'], [19, 'E'], [20, 'E'], [21, 'E']], 0, 0, -1]], 'total_needReceive': 190, 'exe_time': 1024}, '3': {'total_needSend': 210, 'out_links': [['12', 210, [[28, 'E'], [29, 'E'], [30, 'N'], [22, 'N'], [14, 'N']], 0, 0, -1]], 'total_needReceive': 540, 'exe_time': 289}, '4': {'total_needSend': 70, 'out_links': [['3', 70, [[3, 'E'], [4, 'S'], [12, 'S'], [20, 'S']], 0, 0, -1]], 'total_needReceive': 70, 'exe_time': 21}, '5': {'total_needSend': 70, 'out_links': [['3', 70, [[22, 'S'], [30, 'W'], [29, 'W']], 0, 0, -1]], 'total_needReceive': 140, 'exe_time': 12}, '6': {'total_needSend': 40, 'out_links': [['3', 40, [[17, 'S'], [25, 'E'], [26, 'E'], [27, 'E']], 0, 0, -1]], 'total_needReceive': 130, 'exe_time': 188}, '7': {'total_needSend': 90, 'out_links': [['3', 90, [[13, 'W'], [12, 'S'], [20, 'S']], 0, 0, -1]], 'total_needReceive': 70, 'exe_time': 138}, '8': {'total_needSend': 50, 'out_links': [['3', 50, [[42, 'E'], [43, 'E'], [44, 'N'], [36, 'N']], 0, 0, -1]], 'total_needReceive': 60, 'exe_time': 135}, '9': {'total_needSend': 100, 'out_links': [['3', 100, [[55, 'N'], [47, 'N'], [39, 'N'], [31, 'W'], [30, 'W'], [29, 'W']], 0, 0, -1]], 'total_needReceive': 100, 'exe_time': 175}, '10': {'total_needSend': 40, 'out_links': [['3', 40, [[8, 'S'], [16, 'S'], [24, 'E'], [25, 'E'], [26, 'E'], [27, 'E']], 0, 0, -1]], 'total_needReceive': 80, 'exe_time': 128}, '11': {'total_needSend': 80, 'out_links': [['3', 80, [[60, 'N'], [52, 'N'], [44, 'N'], [36, 'N']], 0, 0, -1]], 'total_needReceive': 100, 'exe_time': 167}, '12': {'total_needSend': 0, 'out_links': [], 'total_needReceive': 210, 'exe_time': 122}}
    print('inputfile：', inputfile)
    print('row: ', rowNum)
    task = onlineTimeline(inputfile,8)
    task.loadGraphByDict(TaskGraph11,MapResult,[],[0,0],12)
    task.computeTime()


if __name__ == "__main__":
    main(sys.argv[1:])
    


