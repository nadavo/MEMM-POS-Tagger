from collections import defaultdict
from time import time
from math import ceil, floor
from random import sample
from multiprocessing import Pool
from smtplib import SMTP
from email.message import EmailMessage

"""Constants"""
TAGCHAR = '_'
WHITESPACE = ' '
ENDOFSENTENCE = '.'
BEGIN = '*'
NUM_THREADS = 4
DEFAULT_CUTOFF_FRACT = 0.3333333

"""File names"""
train_file = "data/train.wtag"
test_file = "data/test.wtag"
comp_file = "data/comp.words"
basic_tagged_comp_file = "competition/comp_m1_200689768.wtag"
advanced_tagged_comp_file = "competition/comp_m2_200689768.wtag"


class SimpleDataReader:
    """Class used for reading the competition file"""
    def __init__(self, file):
        self.file = file
        self.sentences = []
        self.word_dict = defaultdict(int)
        self.numbers = 0
        self.cap_no_start = 0
        self.__read_data__()
        self.__make_tuples__()

    def __read_data__(self):
        """main reader function"""
        with open(self.file, 'r') as data:
            for line in data:
                sentence = []
                terms = line.rstrip().split(WHITESPACE)
                for word in terms:
                    if self.isNumberWord(word):
                        self.numbers += 1
                    if word[0].isupper() and len(sentence) > 0:
                        self.cap_no_start += 1
                    self.word_dict[word] += 1
                    sentence.append(word)
                self.sentences.append(tuple(sentence))

    def __make_tuples__(self):
        self.sentences = tuple(self.sentences)

    @staticmethod
    def isNumberWord(word):
        if word.isdigit():
            return True
        elif word.isnumeric():
            return True
        elif word.isdecimal():
            return True
        else:
            for char in ('-', ',', '.', '\/'):
                word = word.replace(char, '')
                if word.isdigit():
                    return True
            return False

    def getWordSet(self):
        return tuple(self.word_dict.keys())

    def getWordDict(self):
        return self.word_dict

    def getWordDictSize(self):
        return len(self.word_dict)

    def getSentences(self):
        return self.sentences

    def getSentencesSize(self):
        return len(self.sentences)

    def getSentenceByIndex(self, index):
        if index < 0 or index >= len(self.sentences):
            raise IndexError
        return self.sentences[index]

    def getTagsByIndex(self, index):
        return ()

    def getWordFreq(self, word):
        return self.getWordDict().get(word, False)

    def getNumbers(self):
        return self.numbers

    def getCapNoStart(self):
        return self.cap_no_start

    def getCapStart(self):
        return self.getSentencesSize()

    def getCaps(self):
        return self.getCapNoStart() + self.getCapStart()


class TaggedDataReader:
    """Class used for reading the training and test files
        hold data structures which contain frequency information on words, tags etc.
        these data structures later will be used to create all features instances
        file parameter expects a string of a path to the data file"""
    def __init__(self, file):
        self.file = file
        self.sentences = []
        self.tags = []
        self.tag_dict = defaultdict(int)
        self.word_dict = defaultdict(int)
        self.word_tag_dict = defaultdict(int)
        self.numbers = 0
        self.cap_no_start = 0
        self.word_suffixes = {}
        self.word_prefixes = {}
        self.__read_data__()
        self.__make_tuples__()
        self.tags_bigrams, self.tags_trigrams = self.__tagsToNgrams__()
        self.sorted_tags_list = sorted(self.tag_dict, key=self.tag_dict.get, reverse=True)

    def __read_data__(self):
        """main reader function which also populates the class data structures"""
        with open(self.file, 'r') as data:
            sentence = []
            tags = []
            for line in data:
                terms = line.rstrip().split(WHITESPACE)
                for term in terms:
                    word_tag = tuple(term.split(TAGCHAR))
                    word = word_tag[0]
                    tag = word_tag[1]
                    self.word_tag_dict[word_tag] += 1
                    self.tag_dict[tag] += 1
                    self.__add_to_word_dict__(word, tag)
                    if self.isNumberWord(word):
                        self.numbers += 1
                    if word[0].isupper() and len(sentence) > 0:
                        self.cap_no_start += 1
                    sentence.append(word)
                    tags.append(tag)
                    if tag == ENDOFSENTENCE:
                        self.sentences.append(tuple(sentence))
                        self.tags.append(tuple(tags))
                        sentence = []
                        tags = []

    def __make_tuples__(self):
        self.sentences = tuple(self.sentences)
        self.tags = tuple(self.tags)

    def __tagsToNgrams__(self):
        """function to create trigrams and bigrams from data"""
        bigrams = defaultdict(int)
        trigrams = defaultdict(int)
        for tags in self.getTags():
            tags = list(tags)
            for i in range(2):
                tags.insert(0, BEGIN)
            for k in range(2, len(tags)):
                trigrams[tuple(tags[k-2:k+1])] += 1
                bigrams[tuple(tags[k-1:k+1])] += 1
        return bigrams, trigrams

    def __add_to_word_dict__(self, word, tag):
        """function used to populate the word frequency data structure"""
        if self.word_dict.get(word, None) is None:
            self.word_dict[word] = defaultdict(int)
        self.word_dict[word][tag] += 1

    def __wordsToSuffixes__(self):
        """function used to create suffixes for all word,tag pairs"""
        suffixes = defaultdict(int)
        for word, tag in self.getWordTagDict():
            for suffix in self.getSuffixesForWord(word):
                suffixes[(suffix, tag)] += 1
        return suffixes

    def __wordsToPrefixes__(self):
        """function used to create prefixes for all word,tag pairs"""
        prefixes = defaultdict(int)
        for word, tag in self.getWordTagDict():
            for prefix in self.getPrefixesForWord(word):
                prefixes[(prefix, tag)] += 1
        return prefixes

    def getSuffixesForWord(self, word):
        """function used to generate suffixes for a given word"""
        suffixes = self.word_suffixes.get(word, False)
        if suffixes is not False:
            return suffixes
        suffixes = []
        if word.isalpha():
            boundary = min(5, len(word))
            for i in range(1, boundary):
                suffixes.append(word[-i:])
        suffixes = tuple(suffixes)
        self.word_suffixes[word] = suffixes
        return suffixes

    def getPrefixesForWord(self, word):
        """function used to generate prefixes for a given word"""
        prefixes = self.word_prefixes.get(word, False)
        if prefixes is not False:
            return prefixes
        prefixes = []
        if word.isalpha():
            boundary = min(5, len(word))
            for i in range(2, boundary):
                prefixes.append(word[:i])
        prefixes = tuple(prefixes)
        self.word_prefixes[word] = prefixes
        return prefixes

    @staticmethod
    def isNumberWord(word):
        if word.isdigit():
            return True
        elif word.isnumeric():
            return True
        elif word.isdecimal():
            return True
        else:
            for char in ('-', ',', '.', '\/'):
                word = word.replace(char, '')
                if word.isdigit():
                    return True
            return False

    def getTagSet(self):
        return tuple(self.tag_dict.keys())

    def getTagSetSize(self):
        return self.getTagDictSize()

    def getWordSet(self):
        return tuple(self.word_dict.keys())

    def getWordDict(self):
        return self.word_dict

    def getTagDict(self):
        return self.tag_dict

    def getTagDictSize(self):
        return len(self.tag_dict)

    def getWordDictSize(self):
        return len(self.word_dict)

    def getSentences(self):
        return self.sentences

    def getTags(self):
        return self.tags

    def getSentencesSize(self):
        return len(self.sentences)

    def getTagsSize(self):
        return len(self.tags)

    def getSentenceByIndex(self, index):
        if index < 0 or index >= len(self.sentences):
            raise IndexError
        return self.sentences[index]

    def getTagsByIndex(self, index):
        if index < 0 or index >= len(self.tags):
            raise IndexError
        return self.tags[index]

    def getWordTagSet(self):
        return tuple(self.word_dict.keys())

    def getWordTagDict(self):
        return self.word_tag_dict

    def getWordTagDictSize(self):
        return len(self.word_tag_dict)

    def getTopNTags(self, n):
        return self.getTopNTagsFromList(self.getSortedTagsList(), n)

    def getTopNTagsForWord(self, word, n):
        tags_dict = self.getWordDict().get(word, False)
        if tags_dict is False:
            return self.getTopNTags(n)
        return self.getTopNTagsFromDict(tags_dict, n)

    def getTopNTagsFromDict(self, tags_dict, n):
        sorted_tags = sorted(tags_dict, key=tags_dict.get, reverse=True)
        return self.getTopNTagsFromList(sorted_tags, n)

    def getTopNTagsFromList(self, sorted_tags, n):
        if n == 1:
            return sorted_tags[0]
        elif n >= len(sorted_tags):
            return sorted_tags
        else:
            return sorted_tags[:n]

    def getSortedTagsList(self):
        return self.sorted_tags_list.copy()

    def getWordFreq(self, word):
        tags_dict = self.getWordDict().get(word, False)
        if tags_dict is not False:
            return sum(tags_dict.values())
        else:
            return 0

    def getNumbers(self):
        return self.numbers

    def getCapNoStart(self):
        return self.cap_no_start

    def getCapStart(self):
        return self.getSentencesSize()

    def getCaps(self):
        return self.getCapNoStart() + self.getCapStart()


class HistoryTuple:
    """Class to create History Tuple objects according to <t-2,t-1,W[i:n],i> definition
    Constructor parameters:
        sequence_id - id of sentence in dataset
        sentence - tuple of words in sentence (in order)
        tags - tuple of tags in sentence (in order)
        index - index of specific word in sentence"""
    def __init__(self, sequence_id, sentence, tags, index):
        if index < 0 or index >= len(sentence):
            raise IndexError
        self.index = index
        self.sequence_id = sequence_id
        self.sentence = sentence
        self.tags = tags
        self.t2, self.t1 = self.__get_previous_tags__(tags)

    def __get_previous_tags__(self, tags):
        """function to retrieve 2 previous tags for word in sentence"""
        if len(self.tags) == 0:
            return None, None
        if self.index == 1:
            return BEGIN, tags[self.index-1]
        elif self.index == 0:
            return BEGIN, BEGIN
        else:
            return tags[self.index-2], tags[self.index-1]

    def getSequenceID(self):
        return self.sequence_id.copy()

    def getSentence(self):
        return self.sentence

    def getTags(self):
        return self.tags

    def getWord(self):
        return self.sentence[self.index]

    def getWordTag(self):
        return self.tags[self.index]

    def getT_2(self):
        return self.t2

    def getT_1(self):
        return self.t1

    def getIndex(self):
        return self.index

    def getTupleKey(self):
        return self.t2, self.t1, self.sequence_id, self.index

    def getTagsTrigram(self):
        return self.t2, self.t1, self.getWordTag()

    def getPossibleTagSet(self, data, cutoff=None, add_common=False):
        """function to return tag set which are possible for a given word,
            according to tags which were observed in the data for that word.
            cutoff parameter determines how many tags will be returned at most.
            add_common determines if tags which were not obeserved for the word will be added to set."""
        full_tag_set_size = data.getTagSetSize()
        if cutoff is None:
            cutoff = ceil(full_tag_set_size * DEFAULT_CUTOFF_FRACT)
        elif cutoff >= full_tag_set_size:
            return data.getTagSet()
        word = self.getWord()
        tags_dict = data.getWordDict().get(word, False)
        if tags_dict is False:
            sorted_tags_list = data.getSortedTagsList()
        else:
            sorted_tags_list = sorted(tags_dict, key=tags_dict.get, reverse=True)
        if data.isNumberWord(word) and "CD" not in sorted_tags_list[:cutoff]:
            sorted_tags_list.insert(0, "CD")
        remainder = cutoff - len(sorted_tags_list)
        if remainder < 0:
            return tuple(sorted_tags_list[:cutoff])
        elif add_common is True and remainder > 0:
            top_candidate_tags = data.getSortedTagsList()
            sorted_tags_set = set(sorted_tags_list)
            candidate_set = set(top_candidate_tags) - sorted_tags_set
            while remainder > 0:
                tag_candidate = top_candidate_tags.pop(0)
                if tag_candidate in candidate_set:
                    sorted_tags_list.append(tag_candidate)
                    remainder -= 1
        return tuple(sorted_tags_list)


class Timer:
    """Simple Timer object which prints elapsed time since its creation"""
    def __init__(self, name):
        self.name = name
        self.__start_time = None
        self.__end_time = None
        self.start()

    def start(self):
        self.__start_time = time()

    def stop(self):
        self.__end_time = time()
        self.__get_elapsed__()

    def __get_elapsed__(self):
        """function to return correctly formatted string according to time units"""
        elapsed = (self.__end_time - self.__start_time)
        unit = "seconds"
        if elapsed >= 3600:
            unit = "minutes"
            hours = elapsed / 3600
            minutes = hours % 60
            hours = floor(hours)
            print(self.name, "took", str(hours), "hours and", "{0:.2f}".format(minutes), unit, "to complete")
        elif elapsed >= 60:
            minutes = floor(elapsed / 60)
            seconds = elapsed % 60
            print(self.name, "took", str(minutes), "minutes and", "{0:.2f}".format(seconds), unit, "to complete")
        else:
            print(self.name, "took", "{0:.2f}".format(elapsed), unit, "to complete")


def binary_search(item_list, item):
    first = 0
    last = len(item_list)-1
    found = False
    while first <= last and not found:
        mid = (first + last)//2
        if item_list[mid] == item:
            found = True
        else:
            if item < item_list[mid]:
                last = mid - 1
            else:
                first = mid + 1
    return found


def missing_binary_search(item_list):
    """utility function used for debugging purposes"""
    first = 0
    last = len(item_list)-1
    while first < last:
        mid = (first + last)//2
        if mid not in item_list:
            return mid
        if mid - first != item_list[mid] - item_list[first]:
            last = mid
        elif last - mid != item_list[last] - item_list[mid]:
            first = mid
        else:
            return None


def split_iterable_to_batches(iterable):
    """function used to split an iterable to N batches according to NUM_THREADS parameter
        used to split a process on a large set to smaller sets which will be run in threads"""
    iterable_length = len(iterable)
    batch_size = int(ceil(iterable_length/NUM_THREADS))
    for i in range(0, iterable_length, batch_size):
        yield iterable[i:i + batch_size]


def split_calculation_to_threads(iterable, func, args):
    """function to split a calculation on an iterable set to seperate NUM_THREADS threads
        used to split gradient calculation and viterbi on entire dataset
        to smaller batches which run in parallel threads"""
    args_list = []
    batches = list(split_iterable_to_batches(iterable))
    for batch in batches:
        temp = list(args)
        temp.insert(0, batch)
        args_list.append(tuple(temp))
    with Pool(NUM_THREADS) as p:
        results = p.starmap(func, args_list)
    return results


def sample_sentences_from_file(file, fraction):
    """utility function to randomly sample a fraction of a data file
        used during development to work on smaller datasets"""
    with open(file, 'r') as f:
        lines = f.readlines()
    new_file_size = ceil(fraction*len(lines))
    rand_lines = sample(lines, new_file_size)
    new_file = file+"_sampled-"+str(new_file_size)+".txt"
    with open(new_file, 'w') as f:
        f.writelines(rand_lines)
    return new_file


def sendEmail(message):
    """utility function to send an email with results from a training run"""
    message_string = '\n'.join(message)
    recipients = ['nadavo@campus.technion.ac.il', 'olegzendel@campus.technion.ac.il']
    msg = EmailMessage()
    msg['Subject'] = 'Finished training and predicting MEMM'
    msg['From'] = 'someserver@technion.ac.il'
    msg['To'] = ', '.join(recipients)
    msg.set_content(message_string)
    sender = SMTP('localhost')
    sender.send_message(msg)
    sender.quit()


def compDataWriter(sentences, predictions, output_file):
    """function to write tagged competition file according to model predictions"""
    assert len(sentences) == len(predictions), "Missing predictions for sentences!"
    lines = list()
    for k in range(len(sentences)):
        assert len(sentences) == len(predictions), "Missing tag predictions for words!"
        sentence = sentences[k]
        tags = predictions[k]
        line_list = [sentence[i]+TAGCHAR+tags[i] for i in range(len(sentence))]
        line = WHITESPACE.join(line_list)
        lines.append(line)
    assert len(lines) == len(sentences), "Missing tagged sentence!"
    with open(output_file, 'w') as file:
        file.write("\n".join(lines))


def generateCompTagging(comp_file, model, viterbi_cutoff):
    """function to generate tagging predictions for competition file, write and validate result file"""
    if model.feature_factory.type == "basic":
        output_file = basic_tagged_comp_file
    elif model.feature_factory.type == "advanced":
        output_file = advanced_tagged_comp_file
    data = SimpleDataReader(comp_file)
    model.predict(data, viterbi_cutoff)
    predictions = model.predictions.get(comp_file)
    compDataWriter(data.getSentences(), predictions, output_file)
    validateTaggedCompFile(comp_file, output_file)


def validateTaggedCompFile(comp_file, tagged_comp_file):
    """function to validate the tagged competition file generated by model,
     is identical to original competition file when removing tags"""
    comp_data = SimpleDataReader(comp_file)
    tagged_comp_data = SimpleDataReader(tagged_comp_file)
    assert comp_data.getSentencesSize() == tagged_comp_data.getSentencesSize(), "Missing Sentences!"
    mistakes = 0
    for i in range(comp_data.getSentencesSize()):
        comp_sentence = comp_data.getSentenceByIndex(i)
        tagged_comp_sentence = tagged_comp_data.getSentenceByIndex(i)
        assert len(comp_sentence) == len(tagged_comp_sentence), "Missing Words in Sentence: " + str(i)
        for k in range(len(comp_sentence)):
            word = comp_sentence[k]
            tagged_word = tagged_comp_sentence[k].split(TAGCHAR)[0]
            if word != tagged_word:
                mistakes += 1
                print("Sentences differ:", word, tagged_word)
    if mistakes == 0:
        print("Files are Identical!")
    else:
        print("Files are NOT Identical!")
