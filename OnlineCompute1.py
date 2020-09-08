import sys
import getopt
import json
import math
#import Queue
import networkx as nx
import pylab
import numpy as np
import logging, sys




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

class onlineTimeline: 
    def __init__(self,inputfile,rowNum):
        self.inputfile =inputfile
        self.totalNum = int(rowNum)*int(rowNum)
        self.rowNum = int(rowNum)
        self.NoClink = []
     

        for i in range(0,self.totalNum):
            link = linklist()
            
            self.NoClink.append(link)
         

        self.sendMatrix = [0]
        self.receiveMatrix = [0]
        self.totalSize=0
        self.exeMatric = [0]
        self.taskGraph = {}
        self.pendTask = []
        self.routeNow = {}
       
        self.stateMatrix=[3]
        self.sendingNow = []
        self.MapResult=[]

        self.nowPri = -1

    

        self.nowTime = 0

        self.fullRouteFromRL={}#key-value表示从task_key到task_value的route全部是由RL计算的
        self.partRouteFromRL=[]#前两个元素i，j表示在计算从task_i到task_j的出边，之后的元素就是json里的route格式，如[0, "S"]，当前位置+下一步移动方向
        self.pendTimes=0#由RL计算的路径导致推迟的次数

    def loadGraphByDict(self,taskGraph1,MapResult1,fullRouteFromRL1,partRouteFromRL1):
        for i in range(1,len(taskGraph1)+1):
            self.sendMatrix.append(taskGraph1[str(i)]['total_needSend'])
            self.receiveMatrix.append(taskGraph1[str(i)]['total_needReceive'])
            self.totalSize = self.totalSize + taskGraph1[str(i)]['total_needSend']
            self.exeMatric.append(taskGraph1[str(i)]['exe_time'])
            self.stateMatrix.append(1000)
            for task in taskGraph1[str(i)]['out_links']:  
                task.append(0)
        self.taskGraph = taskGraph1
        self.MapResult = MapResult1
        self.fullRouteFromRL=fullRouteFromRL1
        self.partRouteFromRL=partRouteFromRL1
        print("task graph loaded++++++++++++++++++++++")
        print("sendMatric",self.sendMatrix)
        print("receiveMatric",self.receiveMatrix)
        print("exeMatric",self.exeMatric)
        self.stateMatrix[1]=1
        print("taskGraph",self.taskGraph)
        print("MapResult",self.MapResult)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        
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
        for i in range(1,len(self.taskGraph)+1):
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
        for i in range(1,len(self.taskGraph)+1):
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
    
        print("total time:",self.nowTime)
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
        #print("this is for checking",route,pri,size)
        #print("checking resout+++++++++++++")
        for rt in route:
            #print(rt,self.NoClink[rt[0]].eList,self.NoClink[rt[0]].wList,self.NoClink[rt[0]].nList,self.NoClink[rt[0]].sList)
            if(rt[1] == 'E' and self.NoClink[rt[0]].eList == 1):
                if( task_source in self.fullRouteFromRL.keys() and task_destination==self.fullRouteFromRL[task_source] ):
                    self.pendTimes+=1
                elif(task_source==self.partRouteFromRL[0] and task_destination==self.partRouteFromRL[1] and rt in self.partRouteFromRL):
                    self.pendTimes+=1
                return False
            elif(rt[1] == 'W' and self.NoClink[rt[0]].wList == 1):
                if( task_source in self.fullRouteFromRL.keys() and task_destination==self.fullRouteFromRL[task_source] ):
                    self.pendTimes+=1
                elif(task_source==self.partRouteFromRL[0] and task_destination==self.partRouteFromRL[1] and rt in self.partRouteFromRL):
                    self.pendTimes+=1
                return False
            elif(rt[1] == 'N' and self.NoClink[rt[0]].nList == 1):
                if( task_source in self.fullRouteFromRL.keys() and task_destination==self.fullRouteFromRL[task_source] ):
                    self.pendTimes+=1
                elif(task_source==self.partRouteFromRL[0] and task_destination==self.partRouteFromRL[1] and rt in self.partRouteFromRL):
                    self.pendTimes+=1
                return False
            elif(rt[1] == 'S' and self.NoClink[rt[0]].sList == 1):
                if( task_source in self.fullRouteFromRL.keys() and task_destination==self.fullRouteFromRL[task_source] ):
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

def main(argv):
    inputfile = ''
    rowNum = ''
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

    print('inputfile：', inputfile)
    print('row: ', rowNum)
    task = onlineTimeline(inputfile,rowNum)
    task.loadGraph()
    #task.computeTime()


if __name__ == "__main__":
    main(sys.argv[1:])
    


