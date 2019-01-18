# File handling libraries
import json
from keras.models import load_model
import pickle

# Data processing
from pandas import DataFrame
import random
import re
import numpy as np
from sklearn.model_selection import train_test_split

# Linear model libraries
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import SGDClassifier

# Deep Learning model libraries
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text
from sklearn.preprocessing import LabelBinarizer

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
sns.set_style("whitegrid")

# Constants
text_file = "./languages_homework_X_train.json.txt"
label_file = "./languages_homework_y_train.json.txt"
PUNCTUATION_REGEX = re.compile(r'([!"#$%&()\'*+,-.:;\/<=>?@[\]^_`{|}~\\])')
DIGITS_REGEX = re.compile(r'\d')


def read_data(X_file, y_file):
    data = []
    with open(X_file, 'r') as examples:
        for line in examples:
            object = json.loads(line)
            data.append(object)

    with open(y_file, 'r') as labels:
        i = 0
        for line in labels:
            object = json.loads(line)
            data[i]['label'] = object['classification']
            i = i + 1
    return data


def preprocess(df):
    df = df.drop_duplicates(subset='text', keep='first') # Removes duplicates
    df = df[df['text'].apply(lambda text:len(text.strip())>0)]  # Removes text with only spaces
    return df


def get_data_for_nltk(df, train=True):
    """Preparing the descriptions and categories lists by assigning True
        to every key (word in a description).
        The goal is to get to this for every description:
        [({'I':True,'love':True,'this':True},'pos'),
        ({'I':True,'dislike':True,'this':True}, 'neg')]"""
    def _make_tokens(row):
        text_tokens = {}
        text = row["text"]
        text = PUNCTUATION_REGEX.sub("", text) # Remove punctuations
        text = DIGITS_REGEX.sub("", text) # Remove digits
        # Both characters and word based model
        for word in text.split():
            text_tokens[word] = True
            for char in word:
                text_tokens[char] = True

        trigrams = zip(text, text[1:], text[2:])
        trigrams_list = [tup[0]+tup[1]+tup[2] for tup in trigrams]
        for t in trigrams_list:
            text_tokens[t] = True

        if train:
            return (text_tokens, row['label'])
        else:
            return text_tokens

    df['data_nltk'] = df.apply(_make_tokens, axis=1)
    tokens = df['data_nltk'].tolist()
    return tokens


def build_learning_curve(classifier, train_data, cv_data):
    m = len(train_data) + len(cv_data)
    # Examples to help plot learning curve
    learning_curve_examples = [ex for ex in range(1000, m, m//10)]
    learning_curve_examples.append(m)
    df = DataFrame([])
    training_error = []
    cv_error = []
    random.shuffle(train_data)
    random.shuffle(cv_data)

    for m in learning_curve_examples:
        print("Evaluating {} examples for learning curve".format(m))
        s1 = int(m * 0.8)  # Collect 80% data from train set
        s2 = int(m * 0.2)  # Collect 20% data from cv set
        train_data_slice = train_data[:s1]
        cv_data_slice = cv_data[:s2]

        classifier.train(train_data_slice)
        training_accuracy = nltk.classify.accuracy(
            classifier, train_data_slice) * 100
        cv_accuracy = nltk.classify.accuracy(
            classifier, cv_data_slice) * 100

        training_error.append(100 - training_accuracy)
        cv_error.append(100 - cv_accuracy)

    df["Learning Examples"] = learning_curve_examples
    df["Training Error"] = training_error
    df["CV Error"] = cv_error

    df.plot(x="Learning Examples", y=["Training Error", "CV Error"], kind="line")
    plt.title('Learning Curve to find Bias and Variance')
    print("Saving Learning curve in ./learningcurve.png")
    plt.savefig("./learningcurve.png")


def linear_model(train_df, cv_df):
    train_tokenized_data = get_data_for_nltk(train_df)
    cv_tokenized_data = get_data_for_nltk(cv_df)

    # Set up SGD Classifier via scikitlearn wrapper
    # log loss makes sure its logistic regression
    classifier = SklearnClassifier(SGDClassifier(loss='log', max_iter=5))
    print("Training Linear Classifier...")
    classifier.train(train_tokenized_data)
    print("Training Done")

    print("Saving Linear Classifier")
    save_pickle(classifier, "linear_model.pickle")
    cv_accuracy = nltk.classify.accuracy(classifier, cv_tokenized_data)
    print("Cross Validation Accuracy of Linear Classifier {}%".format(round(cv_accuracy*100, 2)))
    predictions = classifier.classify_many([x for (x,y) in cv_tokenized_data])

    print("Plotting Learning Curve...")
    build_learning_curve(classifier, train_tokenized_data, cv_tokenized_data)

    print("Saving precision, recall and F-scores in performance.txt")
    with open('performance.txt', 'w') as text_file:
        text_file.write('Linear Model Expected Performance Below:')
        text_file.write('\n')
        text_file.write(classification_report(cv_df['label'].tolist(), predictions))


def get_data_for_keras(df):
    # Use Linear Model first to predict if it is roman alphabet language or not
    # Use this information to build features for neural network
    # Character level information for roman alphabet languages is not useful
    # But is important for languages with other scripts

    print("Loading Linear Model...")
    with open("linear_model.pickle", "rb") as fname:
        linear_classifier = pickle.load(fname)

    def _make_tokens(row):
        text = PUNCTUATION_REGEX.sub("", row['text']) # Remove punctuations
        text = DIGITS_REGEX.sub("", text) # Remove digits

        # Both characters and word based model
        roman_alphabet_languages = ['hr', 'no','sh','nn','id','ceb','gl','ms','vo','la','sl','war',
                                    'pt','en','de','it', 'fr', 'es']
        tokens = ""
        for word in text.split():
            tokens= tokens + word + " "
            is_roman_alphabet_language = row['linear_classifier_label'] in roman_alphabet_languages
            if not is_roman_alphabet_language:
                for char in word:
                    tokens= tokens + char + " "

        trigrams = zip(text, text[1:], text[2:])
        trigrams_list = [tup[0]+tup[1]+tup[2] for tup in trigrams]
        for t in trigrams_list:
            tokens = tokens + t + " "
        return tokens

    data = get_data_for_nltk(df, train=False)
    df['linear_classifier_label'] = linear_classifier.classify_many(data)
    df['tokens'] = df.apply(_make_tokens, axis=1)
    return df


def neural_network(train_df):
    train_df = get_data_for_keras(train_df)
    np.random.seed(7)
    vocab = 50000
    num_labels = len(train_df['label'].unique())
    tokenizer = text.Tokenizer(num_words=vocab)
    tokenizer.fit_on_texts(train_df['tokens'])
    # To convert text sentences to equally sized vectors (binary mode)
    train_data = tokenizer.texts_to_matrix(train_df['tokens'])

    print("Saving Tokenizer in a pickle file...")
    save_pickle(tokenizer, "neural_network_tokenizer.pickle")

    print("Shape of Training Data", train_data.shape)
    # Convert the language categories into one-hot vectors
    encoder = LabelBinarizer()
    encoder.fit(train_df['label'])
    train_data_labels = encoder.transform(train_df['label'])

    print("Shape of Training Data Labels", train_data_labels.shape)

    print("Saving Label Encoder in a pickle file...")
    save_pickle(encoder, "neural_network_label_encoder.pickle")

    # Neural network
    model = Sequential()
    model.add(Dense(512, input_shape=(vocab,), activation='relu'))
    # Dropout to avoid over-fitting
    model.add(Dropout(0.5))
    model.add(Dense(num_labels))
    # Softmax activation is useful for last layer for classifications
    # as it provides a number between 0 and 1 for each category
    model.add(Activation('softmax'))

    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_data, train_data_labels, batch_size=128,
              epochs=2, verbose=1)

    print("Saving the model...")
    model.save('neural_network_language.h5')
    print("Model Saved")


def save_pickle(data, fname):
    with open(fname, "wb") as fname:
        pickle.dump(data, fname)


def read_model(cv_df):
    with open("neural_network_tokenizer.pickle", "rb") as fname:
        tokenizer = pickle.load(fname)

    with open("neural_network_label_encoder.pickle", "rb") as fname:
        encoder = pickle.load(fname)

    model = load_model('neural_network_language.h5')
    cv_df = get_data_for_keras(cv_df)
    cv_data = tokenizer.texts_to_matrix(cv_df['tokens'])

    predictions = model.predict_classes(cv_data)
    labels = encoder.classes_
    predictions_labels = [labels[p] for p in predictions]
    cv_data_labels = cv_df['label'].tolist()
    print(encoder.transform(cv_df['label']))
    score = model.evaluate(cv_data, encoder.transform(cv_df['label']), batch_size=128, verbose=1)
    print("CV Accuracy {}%".format(round(score[1]*100, 2)))

    tuples_results = zip(cv_data_labels, predictions_labels)
    df = DataFrame(columns=["text", "tokens", "actual", "predicted"])
    for i, t in enumerate(tuples_results):
        # Only save if there is a mis-classification
        if t[0] != t[1]:
            df = df.append({"text":cv_df['text'].iloc[i], "tokens":cv_df['tokens'].iloc[i],
                            "actual": t[0], "predicted": t[1]}, ignore_index=True)

    print("Saving information about misclassifications in CV dataset...")
    df.to_csv("misclassifications_analysis.csv")

    print("Saving Confusion Matrix...")
    df2 = DataFrame(confusion_matrix(cv_data_labels, predictions_labels, labels=labels))
    df2.to_csv('confusion_matrix.csv')

    print("Appending precision, recall and F-scores in performance.txt")
    with open('performance.txt', 'a') as text_file:
        text_file.write('\n')
        text_file.write('Neural Network Expected Performance Below:')
        text_file.write('\n')
        text_file.write(classification_report(cv_data_labels, predictions_labels))


def main():
    data = read_data(text_file, label_file)
    df = DataFrame(data, columns=['text', 'label'])   # Converts list of dict to DataFrame
    df = preprocess(df)
    # Split into train and cv sets, stratified so that even languages with lower labels get the same ratio
    train_df, cv_df = train_test_split(df, test_size = 0.20, stratify=df['label'], random_state = 7)
    # linear_model(train_df, cv_df)    # Train Linear Model
     #neural_network(train_df)         # Train Neural Network
    read_model(cv_df)                # Read saved model to analyze performance
    # neural_network(df)               # Having evaluated, final step is to train the model with entire training data set

if __name__ == "__main__":
    main()