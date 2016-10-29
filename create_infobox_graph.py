from infobox_graph import *
import re


filename = 'infoboxEdges.txt'
def create_graph():
	g = Graph()
	times = count = 0
	with open(filename,'r') as theDataset:
		for line in theDataset:
			count += 1
			resources_list = re.findall(r'<http://dbpedia.org/resource/([^>]+)>',line)
			if resources_list == []:
				continue
			# print resources_list
			resource_id_list = []
			for resource in resources_list:
				resource_id_list.append(g.addNode(resource))
			MainResource = resource_id_list[0]
			for resource_ids in resource_id_list[1:]:
				g.addEdge(MainResource,resource_ids)
			if count>100000:
				times += 1
				print times
				count = 0
	# print g.n
	g.calcAllDist()
	return g


g = create_graph()
# print g.n
while(1):
	print "Got the graph"
	topic1 = raw_input("Enter the first resource: ")
	topic2 = raw_input("Enter the second resource: ")
	topic1_Id = g.getId(topic1)
	topic2_Id = g.getId(topic2)
	print "Got the ids " + str(topic1_Id) + " " + str(topic2_Id)
	# print "Distance found = " + str(g.findMinPath(topic1_Id, topic2_Id))
	print "Distance found = " + str(g.dist(topic1_Id, topic2_Id))
