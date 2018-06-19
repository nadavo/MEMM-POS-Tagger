from utils import HistoryTuple, BEGIN, ENDOFSENTENCE
from itertools import product


class ViterbiAlgorithm:
    """Class to create an instance of the Viterbi Algorithm for predicting a given sentence's tags
        instance is initialized for the sentence and run() method populates the tag_sequence list
        which holds the algorithms output.
    Constructor parameters:
        sequence_id - id of sentence in the dataset
        sentence - tuple of words from sentence (in order)
        sentence_tags - tuple of tags from sentence (in order)
        model - trained model object to use for inference
        cutoff - limits the number of possible tags to evaluate for each word in the sequence"""
    def __init__(self, sequence_id, sentence, sentence_tags, model, cutoff=None):
        self.sequence_id = sequence_id
        self.sentence = sentence
        self.sentence_tags = sentence_tags
        self.data = model.data
        if cutoff is None:
            self.cutoff = self.data.getTagSetSize()
        else:
            self.cutoff = cutoff
        self.tags_set = self.data.getTagSet()
        self.prob_func = model.probability
        self.weights = model.getWeights()
        self.pi = {(-1, BEGIN, BEGIN): 1.0}
        self.bp = {}
        self.tag_sequence = []

    def run(self):
        """Main method which runs the algorithm according to implementation learned in lectures and tutorials"""
        sentence_length = len(self.sentence)
        for k in range(sentence_length):
            tag_pairs = tuple(product(self.__calc_possible_tags_set__(k-1), self.__calc_possible_tags_set__(k)))
            for u, v in tag_pairs:
                key = (k, u, v)
                self.pi[key], self.bp[k] = self.__calc_max_probability__(key)
                if self.pi[key] == 0.0000:
                    self.bp[k] = self.data.getTopNTagsForWord(self.sentence[k], 1)
        self.bp[sentence_length], self.bp[sentence_length+1] = self.__calc_last_tags__(sentence_length)
        for k in range(sentence_length):
            self.tag_sequence.append(self.bp.get(k+2, False))

    def __calc_possible_tags_set__(self, index):
        """function to return the possible tag set for a given position in the sentence
            this is an optimization we implemented, instead of running over all possible tags,
            we run over a set of tags which were observed in the data for the given word."""
        if index < 0:
            return tuple(BEGIN)
        if index >= len(self.sentence)-1:
            return tuple(ENDOFSENTENCE)
        return HistoryTuple(self.sequence_id, self.sentence, self.sentence_tags, index).getPossibleTagSet(self.data, self.cutoff, add_common=False)

    def __calc_max_probability__(self, key):
        """function to calculate the maximum probability for each iteration of the main algorithm
            returns the maximum probability and tag for which the probability was maximized"""
        k = key[0]
        u = key[1]
        v = key[2]
        if k < 0:
            return 1.0, BEGIN
        max_pi = 0.00000
        max_bp = None
        possible_tags_set = self.__calc_possible_tags_set__(k-2)
        for t in possible_tags_set:
            new_key = (k-1, t, u)
            history = HistoryTuple(self.sequence_id, self.sentence, self.sentence_tags, k)
            pi_value = self.pi.get(new_key, 0.00000) * self.prob_func(v, history, self.weights)
            if pi_value >= max_pi:
                max_pi = pi_value
                max_bp = t
        return max_pi, max_bp

    def __calc_last_tags__(self, sentence_length):
        """function to return last 2 tags in the sequence from bp data structure"""
        max_pi = 0.0
        max_bp = ()
        tag_pairs = tuple(product(self.__calc_possible_tags_set__(sentence_length-2), self.__calc_possible_tags_set__(sentence_length-1)))
        for u, v in tag_pairs:
            key = (sentence_length-1, u, v)
            pi_value = self.pi.get(key, None)
            if pi_value >= max_pi:
                max_bp = (u, v)
        return max_bp

    def getBestTagSequence(self):
        """method to retrieve the output sequence produced by the Viterbi algorithm"""
        return tuple(self.tag_sequence)
