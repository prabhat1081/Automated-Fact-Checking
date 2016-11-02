try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

import math
import numpy

class Node:
	def __init__(self,topic,ID):
		self.topic = topic
		self.id = ID
		self.degree = 0
		self.connected = []

	def addEdge(self,anotherNodeId):
		self.connected.append(anotherNodeId)
		self.degree+=1

INFINITE = 1000000009
class Graph:
	def __init__(self):
		self.n = 0
		self.nodes = []						# Maintains ID to Node(which contains topic) mapping
		self.dictionary = {}				# Maintains topic to ID mapping
		self.distances = numpy.zeros((1,1))	#The static distances b/w any two nodes will be saved in this matrix. Calc by floyd warshall

	#Creates a Node and returns its ID (Starting from 0). ID is the position of that node in the nodes array in the graph 
	def addNode(self,topic):
		try:	# If already present, return the Id
			return self.dictionary[topic]
		except:
		# if topic not in self.dictionary:
			newNode = Node(topic,self.n)
			self.nodes.append(newNode)
			self.dictionary[topic] = self.n
			self.n+=1
			return self.n-1
		

	def getId(self,topic):
		try:
			return self.dictionary[topic]
		except:
			return -1

	def addEdge(self,aNodeId,anotherNodeId):
		self.nodes[aNodeId].addEdge(anotherNodeId)
		self.nodes[anotherNodeId].addEdge(aNodeId)

	def findMinPath(self,aNodeId,anotherNodeId):
		#Djikstra's Algo
		print self.nodes[aNodeId].topic
		q = Q.PriorityQueue()
		q.put((1,aNodeId))
		while not q.empty():
			intermediateNodeDistance, intermediateNodeId = q.get()
			# print self.nodes[intermediateNodeId].topic
			if intermediateNodeId == anotherNodeId:
				return intermediateNodeDistance
			for nextNodeId in self.nodes[intermediateNodeId].connected:
				q.put((intermediateNodeDistance + math.log(self.nodes[nextNodeId].degree),nextNodeId))

		return INFINITE #INFINITE DISTANCE

	def dist(self,aNodeId,anotherNodeId):
		return self.distances[aNodeId][anotherNodeId]

	def calcAllDist(self):
		#First initialize the distance matrix
		self.distances = numpy.zeros((self.n,self.n))
		for i in range(self.n):
			print i
			for j in range(self.n):
				if j in self.nodes[i].connected:
					self.distances[i][j] = 1
					self.distances[j][i] = 1
				else:
					self.distances[i][j] = INFINITE
					self.distances[j][i] = INFINITE
					
		print "Gonna do floyd Warshall"
		#Floyd Warshall
		for k in range(self.n):
			print k
			for i in range(self.n):
				for j in range(self.n):
					if (self.distances[i][k] + self.distances[k][j]) < self.distances[i][j]:
						self.distances[i][j] = self.distances[i][k] + self.distances[k][j]

		numpy.save("allDistances",self.distances)
