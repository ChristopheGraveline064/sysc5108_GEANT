from cgitb import reset
from collections import defaultdict
from datetime import datetime
import os 
import csv
import sys
import operator
import time
from datetime import datetime

#Initializing the Graph Class
class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}
    
    def addNode(self,value):
        self.nodes.add(value)
    
    def addEdge(self, fromNode, toNode, distance):
        self.edges[fromNode].append(toNode)
        self.distances[(fromNode, toNode)] = distance


dir_tgt = ''
output = '';
num_files = 0;

G_nodes=[]
G_edges=[]
G_distances=[]

cur_time=0.0
done=False
main_traffic_time = []
main_traffic_src = []
main_traffic_dst = []
main_traffic_bw = []

traffic_time = []
traffic_src = []
traffic_dst = []
traffic_bw = []
my_graph =  Graph()
curr_line = 0


for i in range (1,23+1):
    G_nodes.append(i)


#Implementing Dijkstra's Algorithm
def dijkstra(graph, initial):
    visited = {initial : 0}
    path = defaultdict(list)
    nodes = set(graph.nodes)

    while nodes:
        minNode = None
        for node in nodes:
            if node in visited:
                if minNode is None:
                    minNode = node
                elif visited[node] < visited[minNode]:
                    minNode = node
        if minNode is None:
            break

        nodes.remove(minNode)
        currentWeight = visited[minNode]

        for edge in graph.edges[minNode]:
            weight = currentWeight + graph.distances[(minNode, edge)]
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge].append(minNode)
    
    return visited, path

# Generate a graph 
def gen_graph(nodes,edges,distances):
    # gengraph_time = time.time()
    this_graph = Graph()
    for node in nodes:
        this_graph.addNode(node)
    i=0
    for edge in edges:
        this_graph.addEdge(edges[i][0],edges[i][1],distances[i])        
        i=i+1
    # print("--- Gen_Graph Time: %s seconds ---" % (time.time() - gengraph_time))
    return this_graph

# Pull topology data from "graph.txt"
with open(os.path.join(os.getcwd(),'graph.txt'),'r')as f:
    lines = f.readlines()
    for line in lines:
        if '->' in line:
            s_line = str(line).replace(";\n","")
            src =int(s_line.split("->")[0]);
            dst =int(s_line.split("->")[1]);
            G_edges.append((src,dst))
            G_distances.append(float(10e9/1e3)) #10G links, recall traffic is in kb

# Reset link utilizations 
def reset_utilization(graph,link_bw):
    num=0
    for i in G_edges:
        G_distances[num]=link_bw
        num=num+1
    this_graph=gen_graph(G_nodes,G_edges,G_distances)
    return this_graph

# Get utilization of a link
def get_utilization(src,dst):
    i=0
    util=0
    for link in G_edges:
        if (src,dst)==link:
            util=(10e9/1e3-G_distances[i])/(10e9/1e3)
        i=i+1
    return util

# Modifies distance metric between src->dst pair
def sub_distance(src,dst,bw):
    # subdistance_time = time.time()
    num=0
    for i in G_edges:
        if((src,dst)==i):
            # print ("Current BW usage between ",i,G_distances[num])
            G_distances[num]-=bw
            # print ("Updated to ",i,G_distances[num])
        num=num+1
    # print("--- Sub_distnace Time: %s seconds ---" % (time.time() - subdistance_time))
    return

# Since <route> only gives 2nd-last hop, recursively call sub_route until it returns the src, then the list of hops will be all the sub-routes
def sub_route(true_src, true_dst, graph, cur_src, cur_dst, bw):
    # print("sub_routing " , cur_src , "->" , cur_dst)
    # print ("curr src = " , cur_src)
    # print ("curr dst = " , cur_dst)
    # print ("true src = " , true_src)
    # print ("true dst = " , true_dst)

    tmp_graph=graph
    path=[]
    cur_hop=cur_src
    next_hop=0

    results = dijkstra(tmp_graph,cur_src)
    # print("\t",results)
    next_hop=results[1].get(cur_dst).pop()
    # print("Using path " + str(cur_hop) + ", " + str(next_hop))
    # time.sleep(.5)
    if(next_hop == true_src):
        return str(true_src)
    else: 
        return (sub_route(true_src, true_dst, graph, true_src, next_hop, bw)+","+str(next_hop)) 


# Routes between src->dst using Dijekstra's shortest path algorithm
def route(graph,src,dst,bw):
    # print("routing " , src , "->" , dst)
    # route_time = time.time()
    tmp_graph=graph
    path=[]
    cur_hop=src
    next_hop=0
    sroute=sub_route(src,dst,graph,src,dst,bw)+","+str(dst)
    sroute_split=sroute.split(",")
    sroute_hops=[]
    sroute_num_hops=len(sroute_split)-1
    for i in range (sroute_num_hops):
        sroute_hops.append((int(sroute_split[i]),int(sroute_split[i+1])))
        sub_distance(int(sroute_split[i]),int(sroute_split[i+1]),bw)
    # print(sroute)
    # print(sroute_hops)

    # print("SUB ROUTED.")
    # while(next_hop!=cur_hop):
    #     print("Cur hop" , cur_hop , "\tNext hop ", next_hop)
    #     results = dijkstra(tmp_graph,src)
    #     print("\t",results)
    #     next_hop=results[1].get(dst).pop()
    #     print("Using path " + str(cur_hop) + ", " + str(next_hop))
    #     path.append((cur_hop,next_hop))
    #     sub_distance(cur_hop,next_hop,bw)
    #     tmp_graph=gen_graph(G_nodes,G_edges,G_distances)
    #     cur_hop=next_hop
    # path.append((cur_hop,dst))
    # sub_distance(cur_hop,dst,bw)
    gen_graph(G_nodes,G_edges,G_distances)
    # print("--- Route Time: %s seconds ---" % (time.time() - route_time))
    return sroute_hops

# Returns 
def get_linkroute(paths):
    str_links=''
    # print("G_edges = " ,G_edges)
    # print("Paths= ", paths)
    
    for j in range(0,24):
        for k in range (0,24):
            if((j,k)in G_edges): # if this is a valid link (goes in-order)
                # print("checking link", (j,k))
                if((j,k) in paths): # if this is the current link 
                    str_links+="1,"
                    # print("found link.", (j,k))
                else: # unused link, so don't set 
                    str_links+="0,"
                    # print("not link.")
    # print(str_links)
    return str_links

def route_timeseries(traffic_src,traffic_dst,traffic_bw):
    # print("Routing...")
    cnt=0
    my_paths = []
    str_out=''
    for i in traffic_src:
        # print(traffic_src[cnt], "\t->\t", traffic_dst[cnt], "\t" , traffic_bw[cnt] ,"kb")
        # print("\nPrinting link utilization...")
        str_out+=(str(traffic_src[cnt])+","+str(traffic_dst[cnt])+","+str(traffic_bw[cnt]))
        for j in range(0,24):
            for k in range (0,24):
                if((j,k)in G_edges):
                    # print("Link: " , j, "\t->\t", k, "\tutil=\t", get_utilization(j, k)*100 , "%")
                    str_out+=(","+str(get_utilization(j, k)))
        
        this_route=route(my_graph,traffic_src[cnt],traffic_dst[cnt],traffic_bw[cnt])
        my_paths.append(this_route)
        str_this_route=get_linkroute(this_route)
        str_out+=str_this_route

        # print(str_out)
        # for partners in G_edges:    
        #     print("Link: " , partners [0], "\t->\t", partners [1], "\tutil=\t", get_utilization(partners [0], partners [1])*100 , "%")
        str_out+="\n"
        cnt+=1


    link_bw = 10e9/1e3 # 10G link in units of kb
    reset_utilization(my_graph,link_bw) # reset this since solution is supposed to be locally optimal for each time series
    # FROM PAPER:
    # Notice that the solutions provided by MILP are locally optimal, since they provide the optimal allocation of the flow given the active flows at that time. However, this does not guarantee optimal congestion as new flows arrive, since we do not re-optimize the paths for flows already in the network,

    # print("\nResetting utilizations for next run (local optimization)...")
    # for partners in G_edges:
        # print("Link: " , partners [0], "\t->\t", partners [1], "\tutil=\t", get_utilization(partners [0], partners [1])*100 , "%")
    return str_out

# Put dataset into main memory
def pull_dataset():
    global main_traffic_time
    global main_traffic_src
    global main_traffic_dst
    global main_traffic_bw

    # pull traffic data from "data.csv"
    with open ('data.csv') as csv_file:
        csv_reader=csv.reader(csv_file, delimiter=',')
        line_count=0
        for row in csv_reader:
            if(not row):
                pass
            else:
                if(row[1]!=row[2]): # skip local interface (i.e. src->src)
                    main_traffic_time.append(int(float(row[0])))
                    main_traffic_src.append(int(row[1]))
                    main_traffic_dst.append(int(row[2]))
                    main_traffic_bw.append(float(row[3]))
    print("Done pulling dataset.")


def process_timeseries(traffic_src,traffic_dst,traffic_bw,timestamp):
    # use the new method to generate the graph, below:


    # pull traffic data from local memory (traffic_* vars)
    global curr_line
    while (True):
        if(float(main_traffic_time[curr_line])==timestamp):
            if(main_traffic_src[curr_line]!=main_traffic_dst[curr_line]): # skip local interface (i.e. src->src)
                traffic_time.append(int(float(main_traffic_time[curr_line])))
                traffic_src.append(int(main_traffic_src[curr_line]))
                traffic_dst.append(int(main_traffic_dst[curr_line]))
                traffic_bw.append(float(main_traffic_bw[curr_line]))
            curr_line+=1
        else: 
            global cur_time
            cur_time=float(main_traffic_time[curr_line]) # update to next timestamp
            # print("Updated timestamp to ", cur_time)
            return




t=0
my_graph=gen_graph(G_nodes,G_edges,G_distances)
now=datetime.now()

start_time = time.time()
pull_dataset()
dt_string = now.strftime("%d_%m_%Y_%H-%M-%S")
dump_filename='dump_'+dt_string+'.csv'
output_file = open(dump_filename, 'a')

print ("Data will be output into output file:\n\t " + dump_filename)

symbol_src = os.path.join(os.getcwd(),dump_filename)
symbol_dst = os.path.join(os.getcwd(),'dump.csv')

if os.path.exists (symbol_dst):
    print ("Removing old symbolic link...")
    os.remove(symbol_dst)
    print ("\tDone.")

os.symlink(symbol_src,symbol_dst)
print ("Linked the following: \n\t" + symbol_dst + "\n\tto:\n\t" + symbol_src)

while(done==False):
    print("Running timeseries ", cur_time)
    s_time = time.time()
    process_timeseries(traffic_src,traffic_dst,traffic_bw,cur_time)
    output_file.write(route_timeseries(traffic_src,traffic_dst,traffic_bw))
    output_file.write("\n\n\n\n")
    traffic_src=[]
    traffic_dst=[]
    traffic_bw=[]
    print("--- %s seconds ---" % (time.time() - s_time))

output_file.close()
print("--- %s seconds ---" % (time.time() - start_time))
