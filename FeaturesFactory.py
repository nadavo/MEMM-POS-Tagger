import numpy as np
from utils import HistoryTuple
from collections import defaultdict


class FeaturesFactory:
    """Abstract class to be inherited by Basic and Advanced feature factories,
    which create features for each model respectively.
    contains generic methods and data structures used to create features for each model type.
    data parameter is a TaggedDataReader object instance
    cutoff parameter determines what is the minimum frequency in the data for each feature in the vector,
    in order to trim the vector size and remove very rare feature occurrences.
    In practice, there is no numerical feature vector implemented,
    but features_index dictionary holds the index for each feature instance in the vector"""
    def __init__(self, data, cutoff=0):
        self.data = data
        self._cutoff = cutoff
        self.features_dict = defaultdict(int)
        self.histories_dict = {}
        self.null_histories_set = set()
        self._features_index = defaultdict(int)
        self.features_list = []
        self._features_vector_length = len(self._features_index)
        self.empirical_counts = []
        self.feature_freq = defaultdict(int)

    def getCutoffParameter(self):
        return self._cutoff

    def getFeaturesVectorLength(self):
        return self._features_vector_length

    def getFeaturesIndices(self, tag, history, in_data=True):
        """Abstract method - implemented in child classes"""
        pass

    def getFeatureDicts(self):
        """Abstract method - implemented in child classes"""
        pass

    def getFeatureNames(self):
        """Abstract method - implemented in child classes"""
        pass

    def __checkFeatureIndex__(self, index, indexes):
        """function to check if specific feature instance has an index in the features vector"""
        if index is not False:
            indexes.append(index)

    def __return_feature_index__(self, tup):
        """function to return the index of a specific feature instance in the features vector"""
        index = self._features_index.get(tup, False)
        return index

    def __generate_features_index__(self, feature_names, dictionaries):
        """method to populate the features_index dictionary, get the feature vector size
        and list of features instances which made the cutoff,
        according to feature names and respective DataReader dictionaries"""
        keys = []
        for name, dictionary in zip(feature_names, dictionaries):
            features = []
            for feature in dictionary.keys():
                if dictionary.get(feature) > self._cutoff:
                    features.append((name, feature))
                    self.feature_freq[name] += 1
            keys.extend(features)
        for i in range(len(keys)):
            self._features_index[keys[i]] = i
        self.features_list = tuple(keys)
        self._features_vector_length = len(keys)

    def __generate_all_features_indices__(self):
        """method to populate utility dictionaries used for quick feature retrieval during future calculations"""
        features = self.features_dict
        histories = self.histories_dict
        for k in range(self.data.getSentencesSize()):
            sentence = self.data.getSentenceByIndex(k)
            tags = self.data.getTagsByIndex(k)
            for i in range(len(sentence)):
                history = HistoryTuple(k, sentence, tags, i)
                history_key = (tags[i], history.getTupleKey())
                features_indices = self.getFeaturesIndices(tags[i], history, True)
                features_key = tuple(features_indices)
                features[features_key] += 1
                if len(features_indices) == 0:
                    self.null_histories_set.add(history_key)
                histories[history_key] = features_indices

    def __calc_empirical_counts__(self):
        """method to calculate the empirical counts part in the gradient calculation"""
        self.empirical_counts = np.zeros(self._features_vector_length, dtype=float)
        for feature, freq in self.features_dict.items():
            for index in feature:
                self.empirical_counts[index] += freq
        assert len(self.empirical_counts) == np.count_nonzero(self.empirical_counts), "0 in empirical counts vector"

    def getEmpiricalCounts(self):
        """method to return the calculated empirical counts vector for gradient calculation"""
        return self.empirical_counts


class BasicFeatures(FeaturesFactory):
    """Class to implement feature creation for Basic model.
    Inherits from FeatureFactory abstract class"""
    def __init__(self, data, cutoff=0):
        super().__init__(data, cutoff)
        self.type = "basic"
        # very bad OOP but works at this point :(
        if isinstance(self, AdvancedFeatures):
            return
        self.__generateBasicFeaturesIndex__()
        self.__generateAllBasicFeaturesIndices__()
        self.__calc_empirical_counts__()

    def __generateBasicFeaturesIndex__(self):
        self.__generate_features_index__((self.getFeatureNames()), (self.getFeatureDicts()))

    def __generateAllBasicFeaturesIndices__(self):
        self.__generate_all_features_indices__()

    def __f100__(self, word_tag):
        """(Word,Tag) pair Feature"""
        return self.__return_feature_index__(("f100", word_tag))

    def __f103__(self, tag_trigram):
        """Trigram Tags Feature"""
        return self.__return_feature_index__(("f103", tag_trigram))

    def __f104__(self, tag_bigram):
        """Bigram Tags Feature"""
        return self.__return_feature_index__(("f104", tag_bigram))

    def getFeaturesIndices(self, tag, history, in_data=True):
        """method to return the feature vector indices of the specific feature instances for a given HistoryTuple and tag
        in_data parameter is an optimization for calls which we know were not obeserved in the data, to skip checking f100"""
        indices = []
        if in_data:
            self.__checkFeatureIndex__(self.__f100__((history.getWord(), tag)), indices)
        self.__checkFeatureIndex__(self.__f103__((history.getT_2(), history.getT_1(), tag)), indices)
        self.__checkFeatureIndex__(self.__f104__((history.getT_1(), tag)), indices)
        return indices

    def getFeatureDicts(self):
        """method to return relevant TaggedDataReader dictionaries for Basic model features creation"""
        return [self.data.getWordTagDict(), self.data.tags_trigrams, self.data.tags_bigrams]

    def getFeatureNames(self):
        """method to return feature names for Basic model features creation"""
        return ["f100", "f103", "f104"]


class AdvancedFeatures(BasicFeatures):
    """Class to implement feature creation for Advanced model.
        Inherits from BasicFeatures class"""
    def __init__(self, data, cutoff=0):
        super().__init__(data, cutoff)
        self.type = "advanced"
        self.__tags = self.data.getTagDict()
        self.__words = self.data.getWordSet()
        self.__suffixes = self.data.__wordsToSuffixes__()
        self.__prefixes = self.data.__wordsToPrefixes__()
        self.__numbers = {"fNum": self.data.getNumbers()}
        self.__caps = {"fCapStart": self.data.getSentencesSize()}
        self.__caps_no_start = {"fCapNoStart": self.data.getCapNoStart()}
        self.__generateAdvancedFeaturesIndex__()
        self.__generateAllAdvancedFeaturesIndices__()
        self.__calc_empirical_counts__()

    def __f101__(self, suffix_tag):
        """Spelling feature for all suffixes of length <=4"""
        return self.__return_feature_index__(("f101", suffix_tag))

    def __f102__(self, prefix_tag):
        """Spelling feature for all prefixes of length <=4"""
        return self.__return_feature_index__(("f102", prefix_tag))

    def __f105__(self, tag):
        """Unigram Tags Feature"""
        return self.__return_feature_index__(("f105", tag))

    def __fNum__(self, word):
        """Number Feature"""
        if self.data.isNumberWord(word):
            return self.__return_feature_index__(("fNum", "fNum"))
        else:
            return False

    def __fCap__(self, word, position):
        """Words which start with Capital letters by position in sentence Features"""
        if word[0].isupper():
            if position == 0:
                return self.__return_feature_index__(("fCapStart", "fCapStart"))
            elif position > 0:
                return self.__return_feature_index__(("fCapNoStart", "fCapNoStart"))
        else:
            return False

    def __generateAdvancedFeaturesIndex__(self):
        self.__generate_features_index__((self.getFeatureNames()), (self.getFeatureDicts()))

    def __generateAllAdvancedFeaturesIndices__(self):
        self.__generate_all_features_indices__()

    def getFeaturesIndices(self, tag, history, in_data=True):
        """method to return the feature vector indices of the specific feature instances for a given HistoryTuple and tag
        first calls on parent (BasicFeatures) method to retrieve th indices of Basic model features, and add the Advanced"""
        indices = super().getFeaturesIndices(tag, history, in_data)
        word = history.getWord()
        position = history.getIndex()
        for suffix in self.data.getSuffixesForWord(word):
            self.__checkFeatureIndex__(self.__f101__((suffix, tag)), indices)
        for prefix in self.data.getPrefixesForWord(word):
            self.__checkFeatureIndex__(self.__f102__((prefix, tag)), indices)
        self.__checkFeatureIndex__(self.__f105__(tag), indices)
        self.__checkFeatureIndex__(self.__fNum__(word), indices)
        self.__checkFeatureIndex__(self.__fCap__(word, position), indices)
        return indices

    def getFeatureDicts(self):
        """method to return relevant TaggedDataReader dictionaries for Basic model features creation"""
        feature_dicts = super().getFeatureDicts()
        feature_dicts.extend([self.__suffixes, self.__prefixes, self.__tags, self.__numbers, self.__caps, self.__caps_no_start])
        return feature_dicts

    def getFeatureNames(self):
        """method to return feature names for Basic model features creation"""
        feature_names = super().getFeatureNames()
        feature_names.extend(["f101", "f102", "f105", "fNum", "fCapStart", "fCapNoStart"])
        return feature_names
