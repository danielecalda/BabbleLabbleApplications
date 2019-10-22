def compute_perturbations(predicted_labels, true_labels):
    perturbations = []
    for predicted, true in zip(predicted_labels, true_labels):
        if predicted != true:
            perturbations.append(true - predicted)
    return perturbations

def accuracy(predicted_labels, true_labels):
    len_of_examples = len(true_labels)
    correct = 0
    for predicted, true in zip(predicted_labels, true_labels):
        if predicted == true:
            correct += 1
    return percentage(correct, len_of_examples)

def accuracy_lf(predicted_labels, true_labels):
    len_of_examples = len(true_labels)
    correct = 0
    for predicted, true in zip(predicted_labels, true_labels):
        if predicted == true:
            correct += 1
    return percentage(correct, len_of_examples)


def coverage_lf(predicted_labels):
    len_of_examples = len(predicted_labels)
    not_zero = 0
    for predicted in predicted_labels:
        if predicted != 0:
            not_zero += 1
    return percentage(not_zero, len_of_examples)


def percentage(part, whole):
    return 100 * float(part)/float(whole)