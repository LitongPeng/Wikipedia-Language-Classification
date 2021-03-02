# Author: Litong Peng (lp5629)
import math
import pickle
import pandas as pd


# the class for decision tree structure
class DecisionTree:
    __slots__ = ['value', 'left', 'right']

    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


# given a sentence, return the True/False list based on 10 attributes
def transfer_to_t_or_f(sentence):
    s = []
    if ' naar ' in sentence or ' deze ' in sentence or ' ik ' in sentence or ' ons ' in sentence or ' ben ' in \
            sentence or ' meest ' in sentence or ' voor ' in sentence or ' niet ' in sentence or ' met ' in sentence \
            or ' ik ' in sentence or ' het ' in sentence or ' ze ' in sentence or ' hem ' in sentence or ' weten ' in \
            sentence or ' jouw ' in sentence or ' dan ' in sentence or ' ook ' in sentence or ' onze ' in sentence or \
            ' ze ' in sentence or ' er ' in sentence or ' hun ' in sentence or ' zo ' in sentence:
        s.append(True)
    else:
        s.append(False)
    if ' i ' not in sentence and ' it ' not in sentence and ' will ' not in sentence and ' me ' not in sentence and \
            'mine ' not in sentence and ' no ' not in sentence and ' not ' not in sentence and ' he ' not in sentence \
            and ' him ' not in sentence and ' his ' not in sentence and ' she ' not in sentence and ' her ' not in \
            sentence and ' we ' not in sentence and ' our ' not in sentence and ' us ' not in sentence and ' they ' \
            not in sentence and ' them ' not in sentence and ' their ' not in sentence and ' there ' not in sentence \
            and ' here ' not in sentence and ' about ' not in sentence and ' for ' not in sentence and ' with ' not \
            in sentence and ' as ' not in sentence and ' so ' not in sentence and ' to ' not \
            in sentence:
        s.append(True)
    else:
        s.append(False)
    if 'aa' in sentence:
        s.append(True)
    else:
        s.append(False)
    if 'ij' in sentence:
        s.append(True)
    else:
        s.append(False)
    if 'oo' in sentence:
        s.append(True)
    else:
        s.append(False)
    if 'ee' in sentence:
        s.append(True)
    else:
        s.append(False)
    # if 'ng' in sentence:
    #     s.append(True)
    # else:
    #     s.append(False)
    # if 'ieuw' in sentence:
    #     s.append(True)
    # else:
    #     s.append(False)
    if ' de ' in sentence:
        s.append(True)
    else:
        s.append(False)
    if 'en' in sentence:
        s.append(True)
    else:
        s.append(False)
    if ' een ' in sentence:
        s.append(True)
    else:
        s.append(False)
    if ' van ' in sentence:
        s.append(True)
    else:
        s.append(False)
    return s


# return 'nl' or 'en' which shows most often
def PLURALITY_VALUE(parent_examples):
    pe = list(parent_examples.index.values)
    return max(set(pe), key=pe.count)


# whether 'nl' and 'en' have same amount,
# and return the one which shows most often
def same_classification(examples):
    same = False
    classification = 'en'
    en_count = list(examples.index.values).count('en')
    nl_count = list(examples.index.values).count('nl')
    if en_count == 0 or nl_count == 0:
        same = True
    if nl_count > en_count:
        classification = 'nl'
    return same, classification


# calculate the entropy
def entropy(x, y):
    if (x + y) > 0:
        px = x / (x + y)
        py = y / (x + y)
    else:
        return 0
    if px == 0 or px == 1 or py == 0 or py == 1:
        return 0
    return -(px * math.log(px, 2)) - (py * math.log(py, 2))


# return the most importance attribute
def IMPORTANCE(attributes, examples):
    index_list = list(examples.index.values)
    en_count = index_list.count('en')
    nl_count = index_list.count('nl')
    N = examples.shape[0]
    Gain = 0
    A = ''
    B = entropy(nl_count, en_count)
    for a in attributes:
        a_true_count = 0
        a_false_count = 0
        nl_true_count = 0
        nl_false_count = 0
        en_true_count = 0
        en_false_count = 0
        for row in range(N - 1):
            val = examples[a][row]
            index = index_list[row]
            if val:
                a_true_count += 1
                if index == 'nl':
                    nl_true_count += 1
                else:
                    en_true_count += 1
            if not val:
                a_false_count += 1
                if index == 'nl':
                    nl_false_count += 1
                else:
                    en_false_count += 1
        Remainder = ((a_true_count / N) * entropy(nl_true_count, en_true_count)) + (
                (a_false_count / N) * entropy(nl_false_count, en_false_count))
        a_gain = B - Remainder
        if a_gain > Gain:
            Gain = a_gain
            A = a
    return A


# using DECISION-TREE-LEARNING algorithm to build the decision tree
def build_tree(examples, attributes, parent_examples):
    if examples.empty:
        return DecisionTree(PLURALITY_VALUE(parent_examples))
    same, classification = same_classification(examples)
    if same:
        return DecisionTree(classification)
    if not attributes:
        return DecisionTree(PLURALITY_VALUE(examples))
    A = IMPORTANCE(attributes, examples)
    left_false_exs = examples.loc[examples[A] == False]
    right_true_exs = examples.loc[examples[A] == True]
    attributes.remove(A)
    left_subtree = build_tree(left_false_exs, attributes, examples)
    right_subtree = build_tree(right_true_exs, attributes, examples)
    return DecisionTree(A, left_subtree, right_subtree)


# using adaboost algorithm to build hypothesis
def build_adaboost(examples, attributes):
    K = 10
    N = examples.shape[0]
    w = []
    for i in range(N):
        w.append(1 / N)
    ll = list(examples.index.values)
    goal = []
    for l in ll:
        if l == 'nl':
            goal.append(True)
        else:
            goal.append(False)
    hypothesis = []
    for k in range(K):
        stump = []
        min_error = math.inf
        y = []
        a = ''
        for l in range(len(attributes)):
            error = 0
            correct_y = []
            for j in range(N):
                if examples.values[j][l] is goal[j]:
                    correct_y.append(j)
                else:
                    error += w[j]
                if error < min_error:
                    min_error = error
                    y = correct_y
                    a = attributes[l]
        for rows in y:
            w[rows] *= min_error / (1 - min_error + 0.0000001)
        w = normalized_weights(w)
        z = math.log((1 - min_error) / (error + 0.000001), 2)
        stump.append(a)
        stump.append(z)
        hypothesis.append(stump)
        attributes.remove(a)
        attributes.append(a)
    return hypothesis


# calculate the normalized weight
def normalized_weights(weights):
    total_weight = sum(weights)
    w = []
    for weight in weights:
        w.append(weight / total_weight)
    return w


# the main train program
def train(examples, hypothesisOut, dt_or_ada):
    punctuations = "!()-[]{};:\,\”\“<>.?&"
    attributes = ['nl_word','no_en_word','aa', 'ij', 'oo', 'ee', ' de ', 'en', ' een ', ' van ']
    nl_or_en = []
    t_or_f = []
    train_dat = open(examples, 'r')
    for line in train_dat:
        nl_or_en.append(line[:2])
        sentence = line[3:].lower()
        for p in punctuations:
            if p in sentence:
                sentence = sentence.replace(p, '')
        t_or_f.append(transfer_to_t_or_f(sentence))
    data = pd.DataFrame(t_or_f, columns=attributes, index=nl_or_en)
    if dt_or_ada == 'dt':
        tree = build_tree(data, attributes, data)
    elif dt_or_ada == 'ada':
        tree = build_adaboost(data, attributes)
    pickle.dump(tree, open(hypothesisOut, 'wb'))


# walk through the tree
def tree(data, h, row):
    if h.left is None or h.right is None:
        return h.value
    elif not data[h.value][row]:
        return tree(data, h.left, row)
    else:
        return tree(data, h.right, row)


# the predict program using decision tree
def predict_dt(h, file):
    result = []
    test_dat = open(file, 'r')
    punctuations = "!()-[]{};:\,\”\“<>.?&"
    attributes = ['nl_word','no_en_word','aa', 'ij', 'oo', 'ee', ' de ', 'en', ' een ', ' van ']
    nl_or_en = []
    t_or_f = []
    for line in test_dat:
        nl_or_en.append(line[:2])
        sentence = line[3:].lower()
        for p in punctuations:
            if p in sentence:
                sentence = sentence.replace(p, '')
        t_or_f.append(transfer_to_t_or_f(sentence))
    data = pd.DataFrame(t_or_f, columns=attributes)
    for row in range(data.shape[0]):
        t = tree(data, h, row)
        result.append(t)
    print("My decision tree prediction: " + str(result))
    # calculate the accuracy
    right = 0
    for i in range(len(nl_or_en)):
        if result[i] == nl_or_en[i]:
            right += 1
    print("My decision tree accuracy: " + str(right / len(nl_or_en) * 100) + "%")


# the predict program using adaboost
def predict_adaboost(h, file):
    result = []
    test_dat = open(file, 'r')
    punctuations = "!()-[]{};:\,\”\“<>.?&"
    attributes = ['nl_word','no_en_word','aa', 'ij', 'oo', 'ee', ' de ', 'en', ' een ', ' van ']
    nl_or_en = []
    t_or_f = []
    for line in test_dat:
        nl_or_en.append(line[:2])
        sentence = line[3:].lower()
        for p in punctuations:
            if p in sentence:
                sentence = sentence.replace(p, '')
        t_or_f.append(transfer_to_t_or_f(sentence))
    data = pd.DataFrame(t_or_f, columns=attributes)
    for row in range(data.shape[0]):
        sum = 0
        for hypothesis in h:
            if data[hypothesis[0]][row]:
                sum += hypothesis[1] * 1
            else:
                sum += hypothesis[1] * -1
        if sum > 0:
            result.append('en')
        else:
            result.append('nl')
    print("My adaboost prediction: " + str(result))
    # calculate the accuracy
    right = 0
    for i in range(len(nl_or_en)):
        if result[i] == nl_or_en[i]:
            right += 1
    print("My adaboost accuracy: " + str(right / len(nl_or_en) * 100) + "%")


# the main predict program
def predict(hypothesis, file):
    h = pickle.load(open(hypothesis, 'rb'))
    if isinstance(h, DecisionTree):
        predict_dt(h, file)
    else:
        predict_adaboost(h, file)


# the main program
def main():
    train_or_predict = input("please enter 'train' or 'predict'")
    if train_or_predict == 'train':
        examples = input("please enter the training file path")
        hypothesisOut = input("please enter the hypothesis file path")
        learning_type = input("please enter 'decision tree' or 'adaboost'")
        if learning_type == 'decision tree':
            train(examples, hypothesisOut, 'dt')
            print("Done!")
        elif learning_type == 'adaboost':
            train(examples, hypothesisOut, 'ada')
            print("Done!")
    elif train_or_predict == 'predict':
        hypothesis = input("please enter the hypothesis file path you just entered for training")
        file = input("please enter the training file path you just entered for training")
        predict(hypothesis, file)
        print("Done!")


if __name__ == '__main__':
    main()
