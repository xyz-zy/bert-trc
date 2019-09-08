import xml.dom.minidom as dom
import re
import glob, os
import itertools

FILE_DIR = "timeml/"

EXTRA_FILE_DIR = "extra/"

class TimeMLExample(object):
	"""
	A single training/test example for the TimeML dataset.
	"""
	def __init__(self, text, e1_pos, e2_pos, label):
		self.text = text
		self.e1_pos = e1_pos
		self.e2_pos = e2_pos
		self.str_label = label
		self.int_label = None

		self.sentences = None
		self.e1_sentence_num = None
		self.e1_sentence_pos= None
		self.e2_sentence_num = None
		self.e2_sentence_pos = None

	def __str__(self):
		return self.text + "\n" + str(self.e1_pos) + " " + str(self.e2_pos) + " " + self.str_label

class Event(object):
	def __init__(self, eid, cls, sentence, pos_in_sentence):
		self.eid = eid
		self.cls = cls
		self.sentence = sentence
		self.pos_in_sentence = pos_in_sentence
	def __str__(self):
		return self.eid + " " + str(self.pos_in_sentence)

class EventInstance(object):
	def __init__(self, eiid, event, tense, aspect, polarity, pos, sentence, pos_in_sentence):
		self.eiid = eiid
		self.event = event
		self.tense = tense
		self.aspect = aspect
		self.polarity = polarity
		self.pos = pos
		self.sentence = sentence
		self.pos_in_sentence = pos_in_sentence

	def __str__(self):
		return self.eiid + " " + str(self.event)

class TimeX3(object):
	def __init__(self, tid, sentence, pos_in_sentence):
		self.tid = tid
		self.sentence = sentence
		self.pos_in_sentence = pos_in_sentence

	def __str__(self):
		return self.tid + " " + str(self.sentence) + " " + str(self.pos_in_sentence)

class Tlink(object):
	def __init__(self, lid, relType, e1, e2):
		self.lid = lid
		self.relType = relType
		self.e1 = e1
		self.e2 = e2

	def __str__(self):
		pass

class TimeMLFile(object):
	def __init__(self, sentences, events, eventInstances, times, tlinks):
		self.sentences = sentences
		self.events = events
		#print(events.keys())
		self.eventInstances = eventInstances
		self.times = times
		self.tlinks = tlinks
		#print(times.keys())

	def get_element(self, id):
		if id in self.events.keys():
			return self.events[id]
		elif id in self.eventInstances.keys():
			return self.eventInstances[id]
		elif id in self.times.keys():
			return self.times[id]
		else:
			return None

	def make_window(self, text, e1_pos, e2_pos, window_size):
		words = text.split()
		flip = False
		first_word = min(e1_pos, e2_pos)
		second_word = max(e1_pos, e2_pos)
		if first_word != e1_pos:
			flip = True
		start_pos = max(0, first_word - window_size)
		end_pos = min(len(words), second_word + window_size + 1)

		first_word = first_word - start_pos
		second_word = second_word - start_pos

		new_text = words[start_pos:end_pos]
		#print(new_text)

		if second_word - first_word > window_size*2:
			window1_end = first_word + window_size + 1
			window2_start = second_word - window_size

			#print(window1_end, window2_start)

			new_text = new_text[:window1_end] + new_text[window2_start:]

			lost_words = window2_start - window1_end

			second_word -= lost_words

		text = " ".join(new_text)

		if flip:
			return text, second_word, first_word
		else:
			return text, first_word, second_word

	def get_example(self, e1, e2, label, window_size = None):
		sent1 = e1.sentence
		sent2 = e2.sentence
		#print(sent1, sent2)

		example = None
		if sent1 >= len(self.sentences) or sent2 >= len(self.sentences):
			return None

		if sent1 == sent2:
			text = self.sentences[sent1]
			e1_pos = e1.pos_in_sentence
			e2_pos = e2.pos_in_sentence
			if window_size:
				text, e1_pos, e2_pos = self.make_window(text, e1_pos, e2_pos, window_size)
			example = TimeMLExample(text, e1_pos, e2_pos, label)

			example.sentences = [self.sentences[sent1]]
			example.e1_sentence_num = 0
			example.e1_sentence_pos = e1_pos
			example.e2_sentence_num = 0
			example.e2_sentence_pos = e2_pos

		elif sent1 < sent2:
			sents = self.sentences[sent1:sent2+1]
			text = " ".join(sents)

			e1_pos = e1.pos_in_sentence
			e2_pos = sum([len(s.split()) for s in sents[:-1]]) + e2.pos_in_sentence

			if window_size:
				text, e1_pos, e2_pos = self.make_window(text, e1_pos, e2_pos, window_size)

			example = TimeMLExample(text, e1_pos, e2_pos, label)
			#print(len(text.split()), e1_pos, e2_pos)
			example.sentences = sents
			example.e1_sentence_num = 0
			example.e1_sentence_pos = e1.pos_in_sentence
			example.e2_sentence_num = len(sents) - 1
			example.e2_sentence_pos = e2.pos_in_sentence

		elif sent1 > sent2:
			sents = self.sentences[sent2:sent1+1]
			text = " ".join(sents)

			e1_pos = sum([len(s.split()) for s in sents[:-1]]) + e1.pos_in_sentence
			e2_pos = e2.pos_in_sentence
			
			if window_size:
				text, e1_pos, e2_pos = self.make_window(text, e1_pos, e2_pos, window_size)

			example = TimeMLExample(text, e1_pos, e2_pos, label)

			example.sentences = sents
			example.e1_sentence_num = len(sents) - 1
			example.e1_sentence_pos = e1.pos_in_sentence
			example.e2_sentence_num = 0
			example.e2_sentence_pos = e2.pos_in_sentence


		return example

class ExampleLoader(object):

	def __init__(self):
		pass


	def get_text_from_element(self, node):
		if node.nodeType == node.TEXT_NODE:
			if node.data.isspace():
				return ""
			else:
				return node.data.replace("\n", " ")
		else:
			text = ""
			for child in node.childNodes:
				text += " "+ self.get_text_from_element(child)+ " "
			return text

	def process_node(self, node, sentence, sentences, events, times):
		if node.nodeName == "EVENT":
			eid = node.attributes['eid'].value
			cls = node.attributes['class'].value
			sentence_num = len(sentences)
			pos_in_sentence = len(sentence.split())

			event = Event(eid=eid, cls=cls, sentence=sentence_num, pos_in_sentence=pos_in_sentence)
			events[eid] = event

		if node.nodeName == "TIMEX3":
			tid = node.attributes['tid'].value
			type = node.attributes['type'].value
			sentence_num = len(sentences)
			pos_in_sentence = len(sentence.split())
			time = TimeX3(tid=tid, sentence=sentence_num, pos_in_sentence=pos_in_sentence)
			times[tid] = time

	def get_instances(self, instance_elts, event_instances, events, input_file):
		for instance in instance_elts:
			eiid = instance.attributes["eiid"].value
			eventID = instance.attributes["eventID"].value
			tense = instance.attributes["tense"].value
			aspect = instance.attributes["aspect"].value
			polarity = instance.attributes["polarity"].value
			pos = instance.attributes["pos"].value

			if eventID not in events:
				print(eventID, input_file)
				continue

			event = events[eventID]
			sentence = event.sentence
			pos_in_sentence = event.pos_in_sentence

			instance = EventInstance(eiid, event, tense, aspect, polarity, pos, sentence, pos_in_sentence)
			event_instances[eiid] = instance

	def process_s_node(self, node, sentences, this_sentence, events, times):
		assert node.nodeName == "s", "not s node " + node.nodeName
		for c1 in node.childNodes:
			if c1.nodeType == c1.TEXT_NODE and not c1.data.isspace():
				text = c1.data.replace("\n", " ")
				this_sentence += text
				#print(text)
			else:
				self.process_node(c1, this_sentence, sentences, events, times)
				text = self.get_text_from_element(c1)
				this_sentence += " " + text + " "
		split_space = this_sentence.split()
		this_sentence = " ".join(split_space)
		#print(this_sentence)
		sentences.append(this_sentence)
		this_sentence = ""

		return this_sentence

	def process_turn_node(self, node, sentences, this_sentence, events, times):
		children = node.getElementsByTagName("s")
		for child in children:
			this_sentence = self.process_s_node(child, sentences, this_sentence, events, times)

		return this_sentence

	def read_extra_file(self, input_file):
		doc = dom.parse(input_file)
		root = doc.childNodes[0]

		elts = root.getElementsByTagName("TEXT")

		if len(elts) == 0:

			elts = root.getElementsByTagName("BODY")
			assert len(elts) == 1, input_file + str(len(elts))
			body = elts[0]

			elts = body.getElementsByTagName("TEXT")

			if len(elts) == 0:
				elts = body.getElementsByTagName("bn_episode_trans")
				assert len(elts) == 1, input_file
				elts = elts[0].getElementsByTagName("section")
				assert len(elts) == 1, input_file
				elts = elts[0].getElementsByTagName("TEXT")
			
		assert len(elts) == 1, input_file
		text_elt = elts[0]

		sentences = []
		events = {}
		times = {}
		this_sentence = ""
		for node in text_elt.childNodes:
			#print(node.nodeName)
			if node.nodeName == "turn":
				this_sentence = self.process_turn_node(node, sentences, this_sentence, events, times)
			elif node.nodeName == "s":
				this_sentence = self.process_s_node(node, sentences, this_sentence, events, times)
			# children = node.getElementsByTagName("s")
			# for child in children:
			# 	for c1 in child.childNodes:
			# 		if c1.nodeType == c1.TEXT_NODE and not c1.data.isspace():
			# 			text = c1.data.replace("\n", "")
			# 			this_sentence += text
			# 			#print(text)
			# 		else:
			# 			self.process_node(c1, this_sentence, sentences, events, times)
			# 			text = self.get_text_from_element(c1)
			# 			this_sentence += " " + text + " "
			# 	split_space = this_sentence.split()
			# 	this_sentence = " ".join(split_space)
			# 	print(this_sentence)
			# 	sentences.append(this_sentence)
			# 	this_sentence = ""

		event_instances = {}
		instanceElts = root.getElementsByTagName("MAKEINSTANCE")
		self.get_instances(instanceElts, event_instances, events, input_file)

		return TimeMLFile(sentences, events, event_instances, times, None)

	def read_file(self, input_file):
		doc = dom.parse(input_file)
		root = doc.childNodes[0]

		sentences = []

		events = {}
		times = {}

		doc_text = ""
		this_sentence = ""
		for node in root.childNodes:
			if node.nodeType == node.TEXT_NODE and not node.data.isspace():
				text = re.sub(r"\n+", " ", node.data)
				split_space = text.split()
				#print(split_space)
				text = " ".join(split_space)
				#print(text)
				split = re.split(r'([\?\.!]) ', text)
				#print(split)
				while len(split) > 0:
					this_sentence += split.pop(0)
					if len(this_sentence) > 0 and this_sentence[-1] in ["?", ".", "!"]:
						sentences.append(this_sentence)
						#print(this_sentence)
						this_sentence = ""
			else:
				self.process_node(node, this_sentence, sentences, events, times)
				text = self.get_text_from_element(node)
				this_sentence += " " + text + " "

		if len(this_sentence) > 0:
			split = this_sentence.split()
			rejoined = " ".join(split)
			sentences.append(rejoined)

		event_instances = {}
		instanceElts = root.getElementsByTagName("MAKEINSTANCE")
		self.get_instances(instanceElts, event_instances, events, input_file)

		tlinks = []
		tlinkElts = root.getElementsByTagName("TLINK")
		for tlinkElt in tlinkElts:
			if tlinkElt.hasAttribute("relatedToEventInstance") and \
			  tlinkElt.hasAttribute("eventInstanceID"):
				lid = tlinkElt.attributes["lid"].value
				relType = tlinkElt.attributes["relType"].value
				eiid = tlinkElt.attributes["eventInstanceID"].value
				relatedToEventInstance = tlinkElt.attributes["relatedToEventInstance"].value

				if eiid not in event_instances or relatedToEventInstance not in event_instances:
					continue

				tlink = Tlink(lid, relType, event_instances[eiid], event_instances[relatedToEventInstance])
				tlinks.append(tlink)

			if tlinkElt.hasAttribute("eventInstanceID") and \
			  tlinkElt.hasAttribute("relatedToTime"):
				lid = tlinkElt.attributes["lid"].value
				relType = tlinkElt.attributes["relType"].value
				eiid = tlinkElt.attributes["eventInstanceID"].value
				relatedToTime = tlinkElt.attributes["relatedToTime"].value

				if eiid not in event_instances or relatedToTime not in times:
					continue
				tlink = Tlink(lid, relType, event_instances[eiid], times[relatedToTime])
				tlinks.append(tlink)

			if tlinkElt.hasAttribute("timeID") and \
			  tlinkElt.hasAttribute("relatedToEventInstance"):
				lid = tlinkElt.attributes["lid"].value
				relType = tlinkElt.attributes["relType"].value
				tid = tlinkElt.attributes["timeID"].value
				eiid = tlinkElt.attributes["relatedToEventInstance"].value

				if tid not in times or eiid not in event_instances:
					continue
				tlink = Tlink(lid, relType, times[tid], event_instances[eiid])
				tlinks.append(tlink)

			if tlinkElt.hasAttribute("timeID") and \
			  tlinkElt.hasAttribute("relatedToTime"):
				lid = tlinkElt.attributes["lid"].value
				relType = tlinkElt.attributes["relType"].value
				tid = tlinkElt.attributes["timeID"].value
				relatedToTime = tlinkElt.attributes["relatedToTime"].value

				if tid not in times or relatedToTime not in times:
					continue
				tlink = Tlink(lid, relType, times[tid], times[relatedToTime])
				tlinks.append(tlink)

		return TimeMLFile(sentences, events, event_instances, times, tlinks)

	def read_examples(self, input_file):
		file_data = self.read_file(input_file)

		examples = []

		for tlink in file_data.tlinks:
			#print(tlink.lid, tlink.relType, tlink.e1, tlink.e2)
			sent1 = tlink.e1.sentence
			sent2 = tlink.e2.sentence
			#print(sent1, sent2)

			example = None
			if sent1 >= len(file_data.sentences) or sent2 >= len(file_data.sentences):
				continue

			if sent1 == sent2:
				text = file_data.sentences[sent1]
				example = TimeMLExample(text, tlink.e1.pos_in_sentence, tlink.e2.pos_in_sentence, tlink.relType)
			elif sent1 < sent2:
				sents = file_data.sentences[sent1:sent2+1]
				text = " ".join(sents)

				e1_pos = tlink.e1.pos_in_sentence
				e2_pos = sum([len(s.split()) for s in sents[:-1]]) + tlink.e2.pos_in_sentence

				example = TimeMLExample(text, e1_pos, e2_pos, tlink.relType)
				#print(len(text.split()), e1_pos, e2_pos)
			elif sent1 > sent2:
				sents = file_data.sentences[sent2:sent1+1]
				text = " ".join(sents)

				e1_pos = sum([len(s.split()) for s in sents[:-1]]) + tlink.e1.pos_in_sentence
				e2_pos = tlink.e2.pos_in_sentence

				example = TimeMLExample(text, e1_pos, e2_pos, tlink.relType)

			if example:
				examples.append(example)
			#print(example)
		return examples


	def antithetics	(self, all_examples):
		new_exs = []

		for ex in all_examples:
			new_ex = None
			if ex.str_label == "AFTER":
				new_ex = TimeMLExample(ex.text, ex.e2_pos, ex.e1_pos, "BEFORE")
				new_ex.int_label = self.label_list.index("BEFORE")
				new_exs.append(new_ex)

			if ex.str_label == "BEFORE":
				new_ex = TimeMLExample(ex.text, ex.e2_pos, ex.e1_pos, "AFTER")
				new_ex.int_label = self.label_list.index("AFTER")
				new_exs.append(new_ex)
				
			if ex.str_label == "DURING":
				new_ex = TimeMLExample(ex.text, ex.e2_pos, ex.e1_pos, "DURING")
				new_ex.int_label = self.label_list.index("DURING")
				new_exs.append(new_ex)

			if new_ex != None:
				new_ex.sentences = ex.sentences
				new_ex.e1_sentence_num = ex.e2_sentence_num
				new_ex.e1_sentence_pos = ex.e2_sentence_pos
				new_ex.e2_sentence_num = ex.e1_sentence_num
				new_ex.e2_sentence_pos = ex.e1_sentence_pos


		all_examples.extend(new_exs)
				

	def assign_num_labels(self, all_examples):
		labels = set()
		for ex in all_examples:
			labels.add(ex.str_label)
		labels = list(labels)
		labels.sort()
		print(labels)
		print(len(labels))
		self.label_list = labels

		for ex in all_examples:
			ex.int_label = labels.index(ex.str_label)	
			# if ex.int_label < 0 or ex.int_label >= 13:
			# 	print("something wrong")


	def read_examples_from_directory(self, dir_path):
		#os.chdir(dir_path)
		examples_list = []	
		for file in glob.glob(dir_path + "*.tml"):
			#file_path = dir_path + file
			examples = self.read_examples(file)
			examples_list.append(examples)

		all_examples = list(itertools.chain.from_iterable(examples_list))
		#antithetics(all_examples)
		print(len(all_examples))
		self.assign_num_labels(all_examples)
		return all_examples

	def read_example_files(self, dir_path):
		all_files = glob.glob(dir_path + "*.tml")
		train_files = all_files[:-4]
		dev_files = all_files[-4:]

		train_examples_list = []
		for file in train_files:
			examples = self.read_examples(file)
			train_examples_list.append(examples)
		train = list(itertools.chain.from_iterable(train_examples_list))

		dev_examples_list = []
		for file in dev_files:
			examples = self.read_examples(file)
			dev_examples_list.append(examples)
		dev = list(itertools.chain.from_iterable(dev_examples_list))
		self.assign_num_labels(train + dev)

		return train, dev

	def read_dense_examples(self, td_path, extra=False, window_size=None):
		class DenseExample(object):
			def __init__(self, file_name, e1, e2, label):
				self.file_name = file_name
				self.e1 = e1
				self.e2 = e2
				self.label = self.parse_label(label)
			def parse_label(self, label):
				labels = {"a":"AFTER", "b":"BEFORE", "i":"INCLUDES", "ii":"IS_INCLUDED", "s":"SIMULTANEOUS", "v":"VAGUE"}
				return labels[label]

		files_to_exs = {}

		f = open(td_path, "r")

		for line in f.readlines():
			split = line.split()
			ex = DenseExample(split[0], split[1], split[2], split[3])

			if ex.file_name not in files_to_exs:
				files_to_exs[ex.file_name] = [ex]
			else: 
				files_to_exs[ex.file_name].append(ex)

		files = list(files_to_exs.keys())
		print(len(files))
		train_files = files[:-2]
		dev_files = files[-2:]

		train_examples = []
		for file_name in train_files:
			file = self.read_extra_file(EXTRA_FILE_DIR + "/" + file_name + ".tml") \
					if extra \
					else self.read_file(FILE_DIR + "/" + file_name + ".tml")

			for ex in files_to_exs[file_name]:
				e1 = file.get_element(ex.e1)
				e2 = file.get_element(ex.e2)

				if e1 == None or e2 == None:
					#print("oops", file_name, ex.e1, ex.e2)
					continue

				example = file.get_example(e1, e2, ex.label, window_size)

				if not example:
					print("o no")
				else:
					train_examples.append(example)

		self.assign_num_labels(train_examples)

		dev_examples = []
		for file_name in dev_files:
			file = self.read_extra_file(EXTRA_FILE_DIR + "/" + file_name + ".tml") \
					if extra \
					else self.read_file(FILE_DIR + "/" + file_name + ".tml")

			for ex in files_to_exs[file_name]:
				e1 = file.get_element(ex.e1)
				e2 = file.get_element(ex.e2)

				if e1 == None or e2 == None:
					#print("oops", file_name, ex.e1, ex.e2)
					continue

				example = file.get_example(e1, e2, ex.label, window_size)

				if not example:
					print("o no")
				else:
					dev_examples.append(example)

		self.assign_num_labels(dev_examples)
		return train_examples, dev_examples


# def main():
# 	parser = argparse.ArgumentParser()

# 	parser.add_argument("--data_dir", default=None, type=str, required=True)
# 	pass


# if __name__ == "main":
# 	main()
# #	"../timebank_1_2/data/timeml/ABC19980108.1830.0711.tml"