import pickle
import random
import spacy
import progressbar


DATA_FILE1 = 'data/train_examples.pkl'
DATA_FILE2 = 'data/train_labels.pkl'


def extract_token(iteration_number):
    print("Extracting tokens")

    DATA_FILE3 = 'data/tokens/tokens_train_list'  + str(iteration_number) + '.pkl'

    with open(DATA_FILE1, 'rb') as f:
        examples = pickle.load(f)
    with open(DATA_FILE2, 'rb') as f:
        stars = pickle.load(f)

    spacy_nlp = spacy.load('en_core_web_sm')

    for example, perturbation, random_numbers in progressbar.progressbar(zip(examples, perturbations, random_numbers_list)):
        doc = spacy_nlp(example)
        tokenized_sentence = [token.text for token in doc if not token.is_stop and token.is_alpha]

        selected_words, random_numbers = select_random_words(tokenized_sentence, perturbation, random_numbers)
        new_tokens_train_list.append(selected_words)
        new_random_numbers_list.append(random_numbers)

    print(len(new_tokens_train_list))
    print(len(random_numbers_list))
    print(perturbations[0])
    print(random_numbers_list[0])
    print(new_random_numbers_list[0])
    print(tokens_train_list[0])
    print(new_tokens_train_list[0])

    with open(DATA_FILE3, 'wb') as f:
        pickle.dump(new_tokens_train_list, f)

    with open(DATA_FILE4, 'wb') as f:
        pickle.dump(new_random_numbers_list, f)

    print("Done")


def generate_random_sequence_pertubation(tokens, perturbation, random_numbers):
    if perturbation == 0:
        return random_numbers
    else:
        dim_not_change = 5 - perturbation
        list_not_change = random.sample(random_numbers, dim_not_change)
        list_change = random.sample(range(len(tokens)), perturbation)
        return list_not_change + list_change


def select_random_words(tokens, perturbation, random_numbers):
    if len(tokens) >= 5:
        if len(random_numbers) == 0:
            temp_random_numbers = random.sample(range(len(tokens)),5)
        else:
            temp_random_numbers = generate_random_sequence_pertubation(tokens,perturbation,random_numbers)
    else:
        return tokens, [0, 1, 2, 3]
    selected_words = []
    for i in temp_random_numbers:
        selected_words.append(tokens[i])
    return selected_words, temp_random_numbers
