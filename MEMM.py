import pickle
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from utils import split_calculation_to_threads, HistoryTuple, Timer
from Viterbi import ViterbiAlgorithm
from collections import defaultdict, Counter
from tabulate import tabulate


class MEMM:
    """Class to implement an MEMM model as learnt in lectures and tutorials
    Constructor parameters:
        feature_factory - a BasicFeatures or AdvancedFeatures feature factory object created for TaggedDataReader object of training data
        regularizer - lambda parameter used for regularization
        pretrained_weights - option to create model with pretrained weights from cache (used for quick model evaluation)"""
    def __init__(self, feature_factory, regularizer=0, pretrained_weights=None):
        self.data = feature_factory.data
        self.feature_factory = feature_factory
        self.regularizer = float(regularizer)
        self.cache = self.getTrainedWeightsCacheName()
        self.weights = self.__initializeWeights__(pretrained_weights)
        self.train_results = None
        self.predictions = {}
        self.correct_tags = defaultdict(int)
        self.wrong_tags = defaultdict(int)
        self.wrong_tag_pairs = defaultdict(int)
        self.wrong_tags_dicts = {}

    def __initializeWeights__(self, pretrained_weights):
        """method to initialize model weights according to pretrained_weights parameter"""
        weights_vector_length = self.feature_factory.getFeaturesVectorLength()
        weights = np.zeros(weights_vector_length, dtype=float)
        if pretrained_weights is True:
            weights = self.loadTrainedWeights(self.getTrainedWeightsCacheName())
        elif pretrained_weights is not None and type(pretrained_weights) is np.ndarray and len(pretrained_weights) == weights_vector_length:
            weights = pretrained_weights
        return weights

    def getWeights(self):
        return self.weights

    def getFeatures(self, tag, history, in_data=False):
        """method to return list of feature instances indices in features vector
        for given tag and HistoryTuple object.
        checks if this tag,history pair was seen before for faster returns"""
        history_key = (tag, history.getTupleKey())
        if history_key in self.feature_factory.null_histories_set:
            return []
        feature = self.feature_factory.histories_dict.get(history_key, None)
        if feature is None:
            feature = self.feature_factory.getFeaturesIndices(tag, history, in_data)
            if len(feature) == 0:
                self.feature_factory.null_histories_set.add(history_key)
        return feature

    def calc_dot_product(self, features, weights):
        """function to calculate dot product between feature and weights vectosrs
         by summing up values of feature indices in weights vector"""
        total = 0.0
        for index in features:
            total += weights[index]
        return total

    def calcDenominatorBatch(self, history, weights, cutoff=None):
        """function to calculate the sum in the denominator of the probability calculation
        also used in the loss function calculation"""
        full_tag_set_size = self.data.getTagSetSize()
        tag_set = history.getPossibleTagSet(self.data, cutoff, add_common=True)
        remainder = float(full_tag_set_size) - len(tag_set)
        total = 0.0
        for tag in tag_set:
            features = self.getFeatures(tag, history, False)
            if len(features) == 0:
                temp = 1.0
            else:
                temp = np.exp(self.calc_dot_product(features, weights), dtype=float)
            total += temp
        if remainder > 0:
            total += remainder
        if total == 0.0:
            total = 0.0001
        return total

    def calcNominator(self, features, weights):
        """function to calculate the nominator in the probability calculation"""
        if len(features) == 0:
            nominator = 1.0
        else:
            product = self.calc_dot_product(features, weights)
            if product == 0.0:
                nominator = 1.0
            else:
                nominator = np.exp(product, dtype=float)
        return nominator

    def probability(self, tag, history, weights, features=None):
        """function to calculate the probability of a specific tag, given a specific history, features and weights vectors"""
        if features is None:
            features = self.getFeatures(tag, history, True)
        nominator = self.calcNominator(features, weights)
        denominator = self.calcDenominatorBatch(history, weights)
        return float(nominator/denominator)

    def calc_loss(self, weights):
        """function to calculate the loss function value over entire dataset, given a weights vector"""
        timer = Timer("Loss Calculation")
        features_sum = 0.0
        denominators_sum = 0.0
        for k in range(self.data.getSentencesSize()):
            sentence = self.data.getSentenceByIndex(k)
            tags = self.data.getTagsByIndex(k)
            for i in range(len(sentence)):
                history = HistoryTuple(k, sentence, tags, i)
                features_sum += self.calc_dot_product(self.getFeatures(tags[i], history, True), weights)
                denominators_sum += np.log(self.calcDenominatorBatch(history, weights, self.data.getTagSetSize()), dtype=float)
        if self.regularizer == 1.0:
            regularization_sum = np.sum(np.power(weights, 2, dtype=float), dtype=float) / 2.0
        elif self.regularizer != 0.0:
            regularization_sum = self.regularizer * np.sum(np.power(weights, 2, dtype=float), dtype=float) / 2.0
        else:
            regularization_sum = 0.0
        total = regularization_sum + denominators_sum - features_sum
        timer.stop()
        print("Loss:", total)
        return total

    def calc_gradient(self, weights):
        """function to calculate the gradient vector over entire dataset, given a weights vector"""
        timer = Timer("Gradient Calculation")
        empirical_counts = self.feature_factory.getEmpiricalCounts()
        expected_counts_dict = self.calcExpectedCountsDict(weights)
        expected_counts = self.calcExpectedCountsVector(expected_counts_dict)
        if self.regularizer == 1.0:
            regularization_counts = weights
        elif self.regularizer != 0.0:
            regularization_counts = self.regularizer * weights
        else:
            regularization_counts = 0.0
        total = regularization_counts + expected_counts - empirical_counts
        timer.stop()
        print("Average Gradient value:", np.mean(total))
        return total

    def calcExpectedCountsDict(self, weights):
        """method for splitting the entire dataset into batches so expected counts calculation could run in parallel threads.
        utilizes the split_calculation_to_threads function.
        aggregates results from all threads to a combined final dictionary {feature_index: expected_counts_value}"""
        dictionary = Counter(defaultdict(float))
        args = [weights]
        results = split_calculation_to_threads(range(self.data.getSentencesSize()), self.calcExpectedCountsBatch, args)
        for item in results:
            dictionary.update(Counter(item))
        return dictionary

    def calcExpectedCountsBatch(self, iterable, weights):
        """method to run expected counts calculation on a specific batch of the dataset.
        this is the function which runs in each independent thread.
        returns a dictionary of {feature_index: expected_counts_value} for its batch of data"""
        dictionary = defaultdict(float)
        for i in iterable:
            sentence = self.data.getSentenceByIndex(i)
            tags = self.data.getTagsByIndex(i)
            for j in range(len(sentence)):
                history = HistoryTuple(i, sentence, tags, j)
                self.calcExpectedCountsBatchInternal(history, weights, dictionary)
        return dictionary

    def calcExpectedCountsBatchInternal(self, history, weights, dictionary):
        """Internal function which calculates the expected counts for a given sentence"""
        cutoff = self.data.getTagSetSize()
        tag_set = history.getPossibleTagSet(self.data, cutoff, add_common=True)
        for tag in tag_set:
            features = self.getFeatures(tag, history, False)
            if len(features) == 0:
                continue
            probability = self.probability(tag, history, weights, features)
            for index in features:
                dictionary[index] += probability

    def calcExpectedCountsVector(self, dictionary):
        """function to convert ExpectedCounts dictionary to a numpy vector for final result in gradient calculation"""
        indexes = dictionary.keys()
        vector = np.zeros(self.feature_factory.getFeaturesVectorLength(), dtype=float)
        for index in indexes:
            vector[index] = dictionary.get(index, 0.0)
        return vector

    def fit(self, max_iter=100, tolerance=0.001, factr=1e12, save=True):
        """method used for training the model, by passing the loss, gradient calculation functions
        and an initial weights vector to the L-BFGS-B minimizer function.
        returns training results and final weights vector.
        has the option of saving the trained weights and results to a local cache file (pickle)"""
        timer = Timer("Training")
        weights, loss, result = fmin_l_bfgs_b(self.calc_loss, self.weights, self.calc_gradient, pgtol=tolerance, maxiter=max_iter, factr=factr)
        if result.get("warnflag", False) != 0:
            print("Warning - gradient didn't converge within", max_iter, "iterations")
        result['loss'] = loss
        print(result)
        self.train_results = result
        self.weights = weights
        timer.stop()
        if save:
            with open(self.getTrainedWeightsCacheName(), 'wb') as cache:
                pickle.dump({'weights': self.weights, 'train_results': self.train_results}, cache)

    def predict(self, data, cutoff=3):
        """method used for initiating inference method and retrieve predictions for entire dataset"""
        timer = Timer("Inference")
        self.predictions[data.file] = self.predictSplit(data, cutoff)
        timer.stop()

    def predictSplit(self, data, cutoff):
        """method for splitting the entire dataset into batches so inference could run in parallel threads.
        utilizes the split_calculation_to_threads function.
        aggregates results from all threads to a combined final dictionary {sequence_index: predictions} """
        dictionary = {}
        args = [data, cutoff]
        results = split_calculation_to_threads(range(data.getSentencesSize()), self.calcPredictionBatch, args)
        for item in results:
            dictionary.update(item)
        return dictionary

    def calcPredictionBatch(self, iterable, data, cutoff):
        """method to run inference on a specific batch of the dataset (Viterbi on each sentence in batch).
        this is the function which runs in each independent thread.
        returns a dictionary of {sequence_index: predictions} for the batch of data"""
        timer = Timer("Predicting " + str(len(iterable)) + " sentences")
        predictions = {}
        for i in iterable:
            sentence = data.getSentenceByIndex(i)
            tags = data.getTagsByIndex(i)
            viterbi = ViterbiAlgorithm(i, sentence, tags, self, cutoff)
            viterbi.run()
            predictions[i] = viterbi.getBestTagSequence()
        timer.stop()
        return predictions

    def evaluate(self, data, verbose=False):
        """method to evaluate the model's predictions vs truth over entire dataset
        by accuracy measure and confusion matrix for top 10 wrong tags.
        must be called only after predict method, otherwise no predictions will be available for evaluation"""
        assert data.getTagsSize() == len(self.predictions.get(data.file, [])), "Predcitions and truth are not the same length!"
        timer = Timer("Evaluation")
        accuracies = []
        for i in range(data.getTagsSize()):
            truth = data.getTagsByIndex(i)
            prediction = self.predictions.get(data.file).get(i, False)
            accuracies.append(self.accuracy(truth, prediction, verbose))
        avg = np.mean(accuracies)
        minimum = np.min(accuracies)
        maximum = np.max(accuracies)
        med = np.median(accuracies)
        print("Results for", data.file)
        print("Total Average Accuracy:", avg)
        print("Minimal Accuracy:", minimum)
        print("Maximal Accuracy:", maximum)
        print("Median Accuracy:", med)
        self.confusionTable(data.file)
        self.confusionMatrix(data.file)
        timer.stop()
        return data.file, avg, minimum, maximum, med

    def accuracy(self, truth, predictions, verbose=False):
        """function to calculate accuracy for a given sentence and model predictions"""
        assert len(truth) == len(predictions), "Predcitions and truth are not the same length!"
        correct = 0
        for i in range(len(truth)):
            key = truth[i]
            subkey = predictions[i]
            if truth[i] == predictions[i]:
                correct += 1
                self.correct_tags[key] += 1
            else:
                self.wrong_tags[key] += 1
                self.wrong_tag_pairs[(key, subkey)] += 1
                if self.wrong_tags_dicts.get(key, False) is False:
                    self.wrong_tags_dicts[key] = defaultdict(int)
                self.wrong_tags_dicts[key][subkey] += 1
                if verbose:
                    print("Mistake in index", i, "(truth, prediction): ", key, subkey)
        result = float(correct) / len(truth)
        if verbose:
            print("Accuracy:", result)
        return result

    def confusionMatrix(self, file, n=10):
        """function to produce Confusion Matrix for top n wrong tags in model evaluation
        'tabulate' package is only used for printing in nice table format"""
        top_wrong_tags = sorted(self.wrong_tags, key=self.wrong_tags.get, reverse=True)[:n]
        header = top_wrong_tags
        rows = []
        for truth in top_wrong_tags:
            columns = [truth]
            for prediction in top_wrong_tags:
                if truth == prediction:
                    columns.append(self.correct_tags.get(truth))
                else:
                    columns.append(self.wrong_tag_pairs.get((truth, prediction)))
            rows.append(columns)
        print("Confusion Matrix for " + self.feature_factory.type + " model on " + file + " dataset")
        header.insert(0, "Truth \ Predicted")
        print(tabulate(rows, headers=header))

    def confusionTable(self, file, n=10):
        """function to produce Confusion Table for top n wrong tags in model evaluation
        'tabulate' package is only used for printing in nice table format"""
        top_wrong_tags = sorted(self.wrong_tag_pairs, key=self.wrong_tag_pairs.get, reverse=True)[:n]
        header = ("Correct Tag", "Model's Tag", "Frequency")
        rows = []
        for truth, prediction in tuple(top_wrong_tags):
            freq = self.wrong_tag_pairs.get((truth, prediction))
            rows.append((truth, prediction, freq))
        print("Confusion Table for " + self.feature_factory.type + " model on " + file + " dataset")
        print(tabulate(rows, headers=header))

    def getTrainedWeightsCacheName(self):
        """method to retrieve cache file name according to model parameters"""
        prefix = "./cache/"
        parameters = "data-" + str(self.data.getSentencesSize()) + "_features-" + self.feature_factory.type +"_weightSize-"\
                     + str(self.feature_factory.getFeaturesVectorLength()) + "_cutoff-" + str(self.feature_factory.getCutoffParameter()) \
                     + "_regularizer-" + str(self.regularizer)
        suffix = "_trained_weights.pkl"
        return prefix + parameters + suffix

    def loadTrainedWeights(self, file):
        """method to load pretrained weights from a given cache file"""
        with open(file, 'rb') as cache:
            trained = pickle.load(cache)
            weights = trained.get('weights')
        return weights
