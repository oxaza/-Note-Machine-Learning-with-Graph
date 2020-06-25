# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:44:48 2020

@author: Administrator
"""

import snap

###### Basic type
i = snap.TInt(10)
print(i.Val)

print('#'*15)
    
###### Vector type
# Create an empty vector创建一个空向量
v = snap.TIntV()
# Add elements 增加元素
v.Add(1)
v.Add(2)
v.Add(3)
v.Add(4)
v.Add(5)
# Print vector size 输出向量的维度
print(v.Len())
'''out:
5
'''
# Get and set element value 可以访问和设置向量中元素的值
print(v[3])
'''out:
4
'''
v[3] = 2 *v[2]
print(v[3])
'''out:
6
'''
# print vector elements 输出向量的元素
for item in v:
    print(item)
'''out:
1
2
3
6
5    
'''
for i in range(0, v.Len()):
    print(i, v[i])
'''out:
0 1
1 2
2 3
3 6
4 5
'''

print('#'*15)

###### Hash Table Type
# Create an empty tatble
h = snap.TIntStrH()
# Add elements
h[5] = "apple"
h[3] = "tomato"
h[9] = "orange"
h[6] = "banana"
h[1] = "apricot"     
# print table size
print(h.Len()) 
'''out:
5
'''
# Get element value
print("h[3]=", h[3])
'''out:
h[3]= tomato
'''
# Set element value
h[3] = "peach"
print("h[3]=", h[3])
'''out:
h[3]= peach
'''
# print table elements
for key in h:
    print(key, h[key])
'''out:
5 apple
3 peach
9 orange
6 banana
1 apricot
'''
# print KeyId
print(h.GetKeyId(3))

print('-'*15)
      
###### Pair Type
# create a Pair
p = snap.TIntStrPr(1, "one")
# print pair values
print(p.GetVal1())
'''out:
1
'''
print(p.GetVal2())
'''out:
one
'''

print('-'*15)

G1 = snap.TNGraph.New()
G2 = snap.GenRndGnm(snap.PNGraph, 100, 1000)

###### Graph Creation
# create directed graph
G1 = snap.TNGraph.New()
# Add nodes before adding edges
G1.AddNode(1)
G1.AddNode(5)
G1.AddNode(12)

G1.AddEdge(1,5)
G1.AddEdge(5,1)
G1.AddEdge(5,12)

# Create undirected graph, directed network
G2 = snap.TUNGraph.New()
N1 = snap.TNEANet.New()

###### Graph Traversal
# Traverse nodes
for NI in G1.Nodes():
    print("node id %d, out-degree %d, in-degree %d" % (NI.GetId(), NI.GetOutDeg(), NI.GetInDeg()))
'''out:
node id 1, out-degree 1, in-degree 1
node id 5, out-degree 2, in-degree 1
node id 12, out-degree 0, in-degree 1
'''
# Traverse edges
for EI in G1.Edges():
    print("(%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId()))
'''out:
(1, 5)
(5, 1)
(5, 12)
'''
# Traverse edges by nodes
for NI in G1.Nodes():
    for DstNId in NI.GetOutEdges():
        print("edge (%d %d)" % (NI.GetId(), DstNId))
'''out:
edge (1 5)
edge (5 1)
edge (5 12)
'''

###### Graph save and loading
# Save text
snap.SaveEdgeList(G1, "test.txt", "List of edges")
# Load text
G2 = snap.LoadEdgeList(snap.PNGraph,"test.txt",0,1)
# Save binary
FOut = snap.TFOut("test.graph")
G2.Save(FOut)
FOut.Flush()
# Load binary
FIn = snap.TFIn("test.graph")
G4 = snap.TNGraph.Load(FIn)





