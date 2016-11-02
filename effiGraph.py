import networkx as nx 
import re
import matplotlib.pyplot as plt

filename = 'infoboxEdges.txt'
def create_graph():
	g = nx.Graph()
	times = count = 0
	with open(filename,'r') as theDataset:
		for line in theDataset:
			count += 1
			resources_list = re.findall(r'<http://dbpedia.org/resource/([^>]+)>',line)
			if resources_list == []:
				continue
			for resource in resources_list[1:]:
				g.add_edge(resources_list[0],resource)

			if count>10:
				times += 1
				print (times)
				count = 0
				break
	# print g.n
	print(g.graph)
	for e in g.edges():
		print(e)
	return g
g = create_graph()
print(g.graph)
#nx.draw(g)
#plt.savefig("path.png")
