import pickle
from src.PipelineReviewFullRandom.utils import most_frequent, calculate_number_wrong, \
    create_tokens_from_choiced_explanations, percentage, high_coverage_elements, high_correct_elements, intersection,\
    average
from metal import LabelModel

DATA_FILE1 = 'data/data.pkl'
DATA_FILE2 = 'data/labels.pkl'


def analyze_for_tokens(ls, parses, iteration_number, modality='most'):

    DATA_FILE5 = 'data/results/predicted_training_labels'  + str(iteration_number) + '.pkl'
    DATA_FILE6 = 'data/tokens/correct_tokens_list' + str(iteration_number) + '.pkl'
    DATA_FILE7 = 'data/results/summary.txt'
    DATA_FILE8 = 'data/results/predicted_test_labels' + str(iteration_number) + '.pkl'
    DATA_FILE9 = 'data/tokens/tokens_train_list' + str(iteration_number) + '.pkl'
    DATA_FILE10 = 'data/tokens/wrong_tokens_list.pkl'

    with open(DATA_FILE2, 'rb') as f:
        Ys = pickle.load(f)

    with open(DATA_FILE9, 'rb') as f:
        tokens_train_list = pickle.load(f)

    L_train = ls[0].toarray()
    L_test = ls[2].toarray()

    predicted_training_labels = []
    predicted_test_labels = []

    if modality == 'most':
        for line in L_train:
            predicted_training_labels.append(most_frequent(line))

        for line in L_test:
            predicted_test_labels.append(most_frequent(line))
    elif modality == 'avg':
        for line in L_train:
            predicted_training_labels.append(average(line))

        for line in L_test:
            predicted_test_labels.append(average(line))
    elif modality == 'label_agg':
        label_aggregator = LabelModel(5)
        label_aggregator.train(ls[0], n_epochs=50, lr=0.003)
        label_aggregator.score(ls[1], Ys[1])
        predicted_training_labels = label_aggregator.predict(ls[0])
        predicted_test_labels = label_aggregator.predict(ls[2])


    with open(DATA_FILE5, 'wb') as f:
        pickle.dump(predicted_training_labels, f)

    with open(DATA_FILE8, 'wb') as f:
        pickle.dump(predicted_test_labels, f)

    len_wrong_train = calculate_number_wrong(Ys[0], predicted_training_labels)

    len_wrong_test = calculate_number_wrong(Ys[2], predicted_test_labels)

    training_accuracy = percentage(len(Ys[0]) - len_wrong_train, len(Ys[0]))

    test_accuracy = percentage(len(Ys[2]) - len_wrong_test, len(Ys[2]))

    with open(DATA_FILE7, 'a') as f:
        f.write("Iteration number: " + str(iteration_number))
        f.write('\n')
        f.write("Number of wrong in training set: " + str(len_wrong_train))
        f.write('\n')
        f.write("Number of wrong in test set: " + str(len_wrong_test))
        f.write('\n')
        f.write("Training Accuracy: " + str(training_accuracy))
        f.write('\n')
        f.write("Test Accuracy: " + str(test_accuracy))
        f.write('\n')
        f.write('\n')

    L_train_transpose = L_train.T

    over_percentage = high_coverage_elements(L_train_transpose)
    correct_elements, wrong_elements = high_correct_elements(L_train_transpose, Ys)

    intersect = intersection(over_percentage, correct_elements)

    new_explanations = []
    for index in intersect:
        explanation = parses[index].explanation
        new_explanations.append(explanation)

    wrong_explanations = []
    for index in wrong_elements:
        wrong_explanation = parses[index].explanation
        wrong_explanations.append(wrong_explanation)

    token_from_explanations = create_tokens_from_choiced_explanations(new_explanations)
    wrong_token_from_explanations = create_tokens_from_choiced_explanations(wrong_explanations)

    correct_tokens_list = []
    wrong_tokens_list = []

    for tokens_list in tokens_train_list:
        correct_tokens = []
        wrong_tokens = []
        for token in tokens_list:
            if token in token_from_explanations:
                correct_tokens.append(token)
            if token in wrong_token_from_explanations:
                wrong_tokens.append(token)
        correct_tokens_list.append(correct_tokens)
        wrong_tokens_list.append(wrong_tokens)

    with open(DATA_FILE6, 'wb') as f:
        pickle.dump(correct_tokens_list, f)

    with open(DATA_FILE10, 'wb') as f:
        pickle.dump(wrong_tokens_list, f)


def analyze_for_expressions(ls, parses, iteration_number, modality='most'):

    DATA_FILE5 = 'data/results/predicted_training_labels'  + str(iteration_number) + '.pkl'
    DATA_FILE6 = 'data/expressions/correct_expressions_list' + str(iteration_number) + '.pkl'
    DATA_FILE7 = 'data/results/summary.txt'
    DATA_FILE8 = 'data/results/predicted_test_labels' + str(iteration_number) + '.pkl'
    DATA_FILE9 = 'data/expressions/expressions_train_list' + str(iteration_number) + '.pkl'
    DATA_FILE10 = 'data/expressions/wrong_expressions_list.pkl'

    with open(DATA_FILE2, 'rb') as f:
        Ys = pickle.load(f)

    L_train = ls[0].toarray()
    L_test = ls[2].toarray()

    predicted_training_labels = []
    predicted_test_labels = []

    if modality == 'most':
        for line in L_train:
            predicted_training_labels.append(most_frequent(line))

        for line in L_test:
            predicted_test_labels.append(most_frequent(line))
    elif modality == 'avg':
        for line in L_train:
            predicted_training_labels.append(average(line))

        for line in L_test:
            predicted_test_labels.append(average(line))
    elif modality == 'label_agg':
        label_aggregator = LabelModel(5)
        label_aggregator.train(ls[0], n_epochs=50, lr=0.003)
        label_aggregator.score(ls[1], Ys[1])
        predicted_training_labels = label_aggregator.predict(ls[0])
        predicted_test_labels = label_aggregator.predict(ls[2])


    with open(DATA_FILE5, 'wb') as f:
        pickle.dump(predicted_training_labels, f)

    with open(DATA_FILE8, 'wb') as f:
        pickle.dump(predicted_test_labels, f)

    len_wrong_train = calculate_number_wrong(Ys[0], predicted_training_labels)

    len_wrong_test = calculate_number_wrong(Ys[2], predicted_test_labels)

    training_accuracy = percentage(len(Ys[0]) - len_wrong_train, len(Ys[0]))

    test_accuracy = percentage(len(Ys[2]) - len_wrong_test, len(Ys[2]))

    with open(DATA_FILE7, 'a') as f:
        f.write("Iteration number: " + str(iteration_number))
        f.write('\n')
        f.write("Number of wrong in training set: " + str(len_wrong_train))
        f.write('\n')
        f.write("Number of wrong in test set: " + str(len_wrong_test))
        f.write('\n')
        f.write("Training Accuracy: " + str(training_accuracy))
        f.write('\n')
        f.write("Test Accuracy: " + str(test_accuracy))
        f.write('\n')
        f.write('\n')


    L_train_transpose = L_train.T

    over_percentage = high_coverage_elements(L_train_transpose)
    correct_elements, wrong_elements = high_correct_elements(L_train_transpose, Ys)

    intersect = intersection(over_percentage, correct_elements)

    new_explanations = []
    for index in intersect:
        explanation = parses[index].explanation
        new_explanations.append(explanation)

    wrong_explanations = []
    for index in wrong_elements:
        wrong_explanation = parses[index].explanation
        wrong_explanations.append(wrong_explanation)

    print(new_explanations)
    '''
    token_from_explanations = create_tokens_from_choiced_explanations(new_explanations)
    wrong_token_from_explanations = create_tokens_from_choiced_explanations(wrong_explanations)
    
    correct_expressions_list = []
    wrong_expressions_list = []

    for tokens_list in _list:
        correct_tokens = []
        wrong_tokens = []
        for token in tokens_list:
            if token in token_from_explanations:
                correct_tokens.append(token)
            if token in wrong_token_from_explanations:
                wrong_tokens.append(token)
        correct_tokens_list.append(correct_tokens)
        wrong_tokens_list.append(wrong_tokens)

    with open(DATA_FILE6, 'wb') as f:
        pickle.dump(correct_tokens_list, f)

    with open(DATA_FILE10, 'wb') as f:
        pickle.dump(wrong_tokens_list, f)
    '''

