import re



def init(filename):
    """
    This function read the tgff file and
    build computation matrix, communication matrix, rate matrix.
    TGFF is a useful tool to generate directed acyclic graph, tfgg file represent a task graph.
    """
    f = open(filename, 'r')

    #Get hyperperiod
    hyperperiod=int(f.readline().split()[1])
    #print(hyperperiod)
    
    f.readline()
    f.readline()
    f.readline()
    f.readline()

    # Calculate the amount of tasks
    num_of_tasks = 0
    while f.readline().startswith('\tTASK'):
        num_of_tasks += 1
    #print('Number of tasks =',num_of_tasks)

    # Build a communication matrix
    data = [[-1 for i in range(num_of_tasks)] for i in range(num_of_tasks)]
    line = f.readline()
    while line.startswith('\tARC'):
        line = re.sub(r'\bt\d_', '', line)
        i, j, d = [int(s) for s in line.split() if s.isdigit()]
        data[i][j] = d
        line = f.readline()
    #for line in data:
    #    print(line)

    while not f.readline().startswith('# type'):
        pass


    # Build a computation matrix
    comp_cost = {}
    line = f.readline()
    while line.startswith('\t'):
        comp_cost.update({int(line.split()[0]):int(line.split()[1])})
        line = f.readline()
    #for key in comp_cost.keys():
    #    print(key,comp_cost[key],type(key),type(comp_cost[key]))

    

    return [hyperperiod, num_of_tasks, data, comp_cost]

#hyperperiod,num_of_tasks,edges,comp_cost=init('./task graph/N4_test.tgff')
#print(num_of_tasks,edges,comp_cost)