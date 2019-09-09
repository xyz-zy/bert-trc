import xml.dom.minidom as dom
import re
import glob, os
import itertools

scriptpath = os.path.realpath(__file__)
scriptdir = foldername = os.path.dirname(scriptpath)

FILE_DIR =  scriptdir + "/TimeBank1.2/timeml/"

EXTRA_FILE_DIR = scriptdir + "/TimeBank1.2/extra"


TE3_DIR = scriptdir + "/TBAQ-cleaned/"

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
