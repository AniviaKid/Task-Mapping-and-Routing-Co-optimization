import sys
import getopt
import json
import re

class readTaskgraph:
#read nodes and links
#return nodes[name,task size] links[from,to,tramsmission size]
    def __init__(self,inputfile,saveflag):
        self.filename = inputfile
        self.saveflag = int(saveflag)


    def readGraph(self):
        taskfile = open(self.filename+'taskGraph.py','r',encoding='UTF-8')
        exefile = open(self.filename+'execution.py','r')
        mapfile = open(self.filename+'Map.py','r')
        prifile = open(self.filename+'priority.py','r')

        i=0
        taskGraph = {}
        
        for line in taskfile.readlines():
            i=i+1
            row = line.split(' ')
            if(i==1):
                inSum = [0]*(len(row)+1)
            j=0
            outSum = 0
            outlinks = []
            for item in row:
                j=j+1

                a = re.findall(r"\d+\.?\d*",item)
                if(len(a)>0):

                    inSum[j]=inSum[j]+int(a[0])
                    outSum = outSum + int(a[0])
                    outlinks.append([str(j),int(a[0])])
                    #print(a)
            taskGraph[str(i)]={'total_needSend':outSum,'out_links':[]}

        #print(inSum)
        

        i=0
        for line in exefile.readlines():
            i=i+1
            taskGraph[str(i)]['total_needReceive']=inSum[i]
            taskGraph[str(i)]['exe_time']=int(line)
        #print(taskGraph)

        MapResult=[-1]
        for line in mapfile.readlines():
            try:
                MapResult.append(int(line))
            except ValueError:
                continue

        print(MapResult)

        i=0
     
        for line in prifile.readlines():
            i=i+1
            #'outlinks':outlinks,
            try:
                source,dest,volumn,route,start_time,end_time,priority = self.analyseLine(line)

                taskGraph[str(source)]['out_links'].append([str(dest),int(volumn),route,int(start_time),int(end_time),int(priority)])
            except KeyError:
                continue
        
        print(taskGraph)  
        if(self.saveflag==1):
            self.writeGraph(taskGraph,MapResult)


    def analyseLine(self,line):
        p1 = re.compile(r'[[](.*?)[]]', re.S)  #最小匹配
        p2 = re.compile(r'[[](.*)[]]', re.S)   #贪婪匹配
        route = re.findall(p2,line)[0]
        routes = re.findall(p1,route)
        #print(route)
        route1 = []
        for item in routes:
            #print("this is item",item)
            items = item.split(',')
            route1.append([int(items[0]),items[1][1]])
        #print(line)
        
        a = line.split(',')
        source = a[0]
        dest = a[1]
        volumn = a[2]
        priority = a[-1]
        end_time = a[-2]
        start_time = a[-3]
        #print(source,dest,volumn,route1,start_time,end_time,priority)
        return source,dest,volumn,route1,start_time,end_time,priority


    def writeGraph(self,taskGraph,MapResult):
        with open(self.filename+"taskGraph.json","w") as f:
            json.dump(taskGraph,f)
            print("Saving taskGraph in to,",self.filename+"taskGraph.json","complete")
        with open(self.filename+"MapResult.json","w") as f:
            json.dump(MapResult,f)
            print("Saving MapResult in to,",self.filename+"MapResult.json","complete")







def main(argv):
    inputfile = ''
    saveflag = ''
    try:
        opts, args = getopt.getopt(argv,"hi:s:",["ifile=","saveflag="])
    except getopt.GetoptError:
        print('Error readTaskgraph.py -i <inputfile> -s <saveflag>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('readTaskgraph.py -i <inputfile> -s <saveflag>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            print("read input file")
            inputfile = arg
        elif opt in ("-s", "--saveflag"):
            saveflag = arg

    print('inputfile: ', inputfile)
    print('saveflag: ', saveflag)
    Graph = readTaskgraph(inputfile,saveflag)
    Graph.readGraph()
    

if __name__ == "__main__":
    main(sys.argv[1:])
    
