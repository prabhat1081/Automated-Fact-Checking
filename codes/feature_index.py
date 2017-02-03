import extractors.tokenizer as tk
import extractors.entity_type
import pprint

text = ""
while(text != "exit"):

	text = input()
	parsed = tk.ner(text)


	pp = pprint.PrettyPrinter(indent=4)

	pp.pprint(parsed)


