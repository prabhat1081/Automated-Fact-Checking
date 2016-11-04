import json
import pandas as pd
import os



dataset = None

basepath = "/home/bt1/13CS10060/btp"


def initialize_map():
	filename = os.path.join(basepath, "ayush_dataset", "annotated_single_all.tsv")
	return pd.read_csv(filename, sep="\t")



def get_sentence(index):
	global dataset
	if(dataset is None):
		dataset = initialize_map()

	did, bid, sid = index

	

	result = dataset[dataset.DebateId == did]
	result = result[result.ID == bid]
	result = result[result.Id_1 == sid]
	return result.iloc[0]['Sentence']
	#result = dataset.query('"DebateId"==@did and "ID"==@bid and "Id-1"==@sid')
	#return result['Sentence']


print(get_sentence((11,2,0)))

