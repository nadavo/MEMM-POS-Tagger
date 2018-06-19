from utils import TaggedDataReader, Timer
from MEMM import HistoryTuple, MEMM
from FeaturesFactory import BasicFeatures, AdvancedFeatures
from Viterbi import ViterbiAlgorithm

"""main file which was mainly used for our personal usage during development.
contains interfaces for different operations: training, predicting, evaluating and generating competition files for models."""

train_file = "data/train.wtag"
test_file = "data/test.wtag"
mini_train_file = "data/train.wtag_mini"


def trainAndPredictModel(model_type="basic", features_cutoff=3, regularizer=1, pretrained=False, viterbi_cutoff=20):
    """main interface method for easily training a model, running inference for predictions,
    evaluate it and generate competition file for it."""
    data = readData(train_file)
    features = constructFeatures(model_type, data, features_cutoff)
    model = createModel(data, features, regularizer, pretrained=pretrained)
    trainModel(model, pretrained=pretrained)
    results = evaluateModel(data, model, viterbi_cutoff)
    results.append("Features Cutoff: " + str(features_cutoff))
    results.append("Regularizer: " + str(regularizer))
    results.append("Viterbi Cutoff: " + str(viterbi_cutoff))


def readData(file):
    timer = Timer("Data Reader")
    data = TaggedDataReader(file)
    timer.stop()
    print("Number of unique tags in data:", data.getTagDictSize())
    print("Number of unique words in data:", data.getWordDictSize())
    print("Number of unique word,tag pairs in data:", data.getWordTagDictSize())
    print("Number of unique trigrams in data:", len(data.tags_trigrams))
    print("Number of unique bigrams in data:", len(data.tags_bigrams))
    print("Number of sentences in data:", data.getSentencesSize())
    print("Number of tag sequences in data:", data.getTagsSize())
    return data


def constructFeatures(complexity, data, features_cutoff):
    timer = Timer("Features Construction"+"-"+complexity)
    if complexity == 'advanced':
        features = AdvancedFeatures(data, features_cutoff)
    elif complexity == 'basic':
        features = BasicFeatures(data, features_cutoff)
    timer.stop()
    print("Features Vector Length:", features.getFeaturesVectorLength())
    print("Number of unique feature index lists in data:", len(features.features_dict))
    print("Maximum frequency in features index lists:", max(features.features_dict.values()))
    print("Feature type frequencies:", features.feature_freq)
    print("Number of unique tag,history pairs in data:", len(features.histories_dict))
    print("Empirical Counts Vector Length:", len(features.empirical_counts))
    return features


def createModel(data, features, regularizer, pretrained=None):
    timer = Timer("Model Creation")
    model = MEMM(features, regularizer, pretrained)
    timer.stop()
    history = HistoryTuple(0, data.getSentenceByIndex(0), data.getTagsByIndex(0), 2)
    timer = Timer("Probability Calculation")
    probs = model.probability('VBZ', history, model.getWeights())
    timer.stop()
    print("Test Probability:", probs)
    # model.calc_loss(model.getWeights())
    # gradient = model.calc_gradient(model.getWeights())
    # print("Gradient length:", len(gradient))
    return model


def trainModel(model, pretrained=None):
    if pretrained is not True:
        model.fit()


def evaluateModel(data, model, viterbi_cutoff):
    timer = Timer("Viterbi Calculation")
    viterbi = ViterbiAlgorithm(0, data.getSentenceByIndex(0), data.getTagsByIndex(0), model, viterbi_cutoff)
    viterbi.run()
    timer.stop()
    print("Truth:", data.getTagsByIndex(0))
    print("Predictions:", viterbi.getBestTagSequence())
    model.accuracy(data.getTagsByIndex(0), viterbi.getBestTagSequence(), True)
    results = list()
    model.predict(data, viterbi_cutoff)
    results.append(str(model.evaluate(data)))
    test_data = TaggedDataReader(test_file)
    model.predict(test_data, viterbi_cutoff)
    results.append(str(model.evaluate(test_data)))
    return results


def mainModel(pretrained, features_cutoff, regularizer, viterbi_cutoff):
    global_timer = Timer("Main Model Run")
    timer = Timer("Data Reader")
    data = TaggedDataReader(train_file)
    timer.stop()
    print("Number of unique tags in data:", data.getTagDictSize())
    print("Number of unique words in data:", data.getWordDictSize())
    print("Number of unique word,tag pairs in data:", data.getWordTagDictSize())
    print("Number of unique trigrams in data:", len(data.tags_trigrams))
    print("Number of unique bigrams in data:", len(data.tags_bigrams))
    print("Number of sentences in data:", data.getSentencesSize())
    print("Number of tag sequences in data:", data.getTagsSize())
    timer = Timer("Features Construction")
    features = BasicFeatures(data, features_cutoff)
    timer.stop()
    print("Features Vector Length:", features.getFeaturesVectorLength())
    # init_weights = initModel(train_file, 0.01)
    # exit(0)
    timer = Timer("Model Creation")
    model = MEMM(features, regularizer=regularizer, pretrained_weights=pretrained)
    timer.stop()
    history = HistoryTuple(0, data.getSentenceByIndex(0), data.getTagsByIndex(0), 2)
    timer = Timer("Probability Calculation")
    probs = model.probability('VBZ', history, model.getWeights())
    timer.stop()
    print("Test Probability:", probs)
    print("Number of unique feature index lists in data:", len(features.features_dict))
    print("Maximum frequency in features index lists:", max(features.features_dict.values()))
    print("Number of unique tag,history pairs in data:", len(features.histories_dict))
    print("Empirical Counts Vector Length:", len(features.empirical_counts))
    model.calc_loss(model.getWeights())
    gradient = model.calc_gradient(model.getWeights())
    print("Gradient length:", len(gradient))
    if pretrained is False:
        model.fit()
    timer = Timer("Viterbi Calculation")
    viterbi = ViterbiAlgorithm(0, data.getSentenceByIndex(0), data.getTagsByIndex(0), model, viterbi_cutoff)
    viterbi.run()
    timer.stop()
    print("Truth:", data.getTagsByIndex(0))
    print("Predictions:", viterbi.getBestTagSequence())
    model.accuracy(data.getTagsByIndex(0), viterbi.getBestTagSequence(), True)
    results = list()
    model.predict(data, viterbi_cutoff)
    results.append(str(model.evaluate(data)))
    test_data = TaggedDataReader(test_file)
    model.predict(test_data, viterbi_cutoff)
    results.append(str(model.evaluate(test_data)))
    global_timer.stop()
    return results


def trainBasicModel(features_cutoff, regularizer, viterbi_cutoff):
    global_timer = Timer("Training Run")
    pretrained = False
    results = mainModel(pretrained, features_cutoff, regularizer, viterbi_cutoff)
    results.append("Viterbi Cutoff: " + str(viterbi_cutoff))
    results.append("Features Cutoff: " + str(features_cutoff))
    results.append("Regularizer: " + str(regularizer))
    global_timer.stop()


def testBasicModel(features_cutoff, regularizer, viterbi_cutoff):
    global_timer = Timer("Test Run")
    pretrained = True
    results = mainModel(pretrained, features_cutoff, regularizer, viterbi_cutoff)
    results.append("Viterbi Cutoff: " + str(viterbi_cutoff))
    results.append("Features Cutoff: " + str(features_cutoff))
    results.append("Regularizer: " + str(regularizer))
    global_timer.stop()


def evaluateBasicModel(features_cutoff, regularizer, viterbi_cutoff):
    pretrained = True
    data = readData(train_file)
    features = constructFeatures("basic", data, features_cutoff)
    model = createModel(data, features, regularizer, pretrained=pretrained)
    #model.weights = model.loadTrainedWeights("./cache/data-4962_cutoff-0_regularizer-1.0_trained_weights.pkl")
    evaluateModel(data, model, viterbi_cutoff)


def evaluateAdvancedModel(features_cutoff, regularizer, viterbi_cutoff):
    pretrained = True
    data = readData(train_file)
    features = constructFeatures("advanced", data, features_cutoff)
    model = createModel(data, features, regularizer, pretrained=pretrained)
    #model.weights = model.loadTrainedWeights("./cache/data-4962_features-advanced_weightSize-14056_cutoff-2_regularizer-1.0_trained_weights.pkl")
    evaluateModel(data, model, viterbi_cutoff)
    print(model.wrong_tags)


def main():
    global_timer = Timer("Training + Test Run")
    viterbi_cutoff = 20
    features_cutoff = 3
    regularizer = 1
    # evaluateBasicModel(features_cutoff, regularizer, viterbi_cutoff)
    # features_cutoff = 3
    # evaluateAdvancedModel(features_cutoff, regularizer, viterbi_cutoff)
    # exit(0)
    pretrained = False
    #trainBasicModel(features_cutoff, regularizer, viterbi_cutoff)
    #testBasicModel(features_cutoff, regularizer, viterbi_cutoff)
    data = readData(train_file)
    features = constructFeatures("advanced", data, features_cutoff)
    model = createModel(data, features, regularizer, pretrained=pretrained)
    trainModel(model, pretrained=pretrained)
    results = evaluateModel(data, model, viterbi_cutoff)
    results.append("Features Cutoff: " + str(features_cutoff))
    results.append("Regularizer: " + str(regularizer))
    results.append("Viterbi Cutoff: " + str(viterbi_cutoff))
    global_timer.stop()


if __name__ == '__main__':
    main()
