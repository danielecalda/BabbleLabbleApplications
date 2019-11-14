from collections import Counter


def most_frequent(line):
    filtered_list = list(filter(lambda a: a != 0, line))
    if len(filtered_list) > 0:
        occurence_count = Counter(filtered_list)
        return occurence_count.most_common(1)[0][0]
    else:
        return 0


def percentage(part, whole):
    return 100 * float(part)/float(whole)


def create_tokens_from_choiced_explanations(explanations):
    tokens_from_explanations = []
    for explanation in explanations:
        word = explanation.word
        label = explanation.label
        new = (word, label)
        tokens_from_explanations.append(new)
    return tokens_from_explanations


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def calculate_number_wrong(real, predicted):
    len_wrong = 0
    for right, wrong in zip(real, predicted):
        if int(right) != wrong:
            len_wrong += 1
    return len_wrong


def high_coverage_elements(l_train):
    over_percentage = []
    for i, line in enumerate(l_train):
        count = 0
        for element in line:
            if element != 0:
                count += 1
        coverage = percentage(count, len(line))
        if coverage > 1:
            over_percentage.append(i)
            # print('Coverage of element number: ' + str(i) + ' is ' + str(coverage))
    print('number of over percentage is: ' + str(len(over_percentage)))
    return over_percentage


def high_correct_elements(l_train, ys):
    correct_elements = []
    wrong_elements = []
    for i, line in enumerate(l_train):
        abstain = 0
        correct = 0
        wrong = 0
        for element, label in zip(line, ys[0]):
            if element == 0:
                abstain += 1
            else:
                if element == label:
                    correct += 1
                else:
                    wrong += 1
        if correct > wrong:
            correct_elements.append(i)
        if wrong > 2*correct:
            wrong_elements.append(i)
            #print('abstain: ' + str(abstain) + ' and correct: ' + str(correct) + ' and wrong: ' + str(wrong))
    print('number of correct elements is: ' + str(len(correct_elements)))
    return correct_elements, wrong_elements