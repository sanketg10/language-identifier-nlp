import json
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import SGDClassifier
import random

import re

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
sns.set_style("whitegrid")

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import numpy as np

import pickle

text_file = "./languages_homework_X_train.json.txt"
label_file = "./languages_homework_y_train.json.txt"
PUNCTUATION_REGEX = re.compile(r'([!"#$%&()\'*+,-.:;\/<=>?@[\]^_`{|}~\\])')
DIGITS_REGEX = re.compile(r'\d')
ALPHABETS_REGEX = re.compile(r'[abcdefghijklmnopqrstuvwxyz]', re.I)

with open("linear_model.pickle", "rb") as fname:
    linear_classifier = pickle.load(fname)

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
        # text_tokens = {char: True for word in row['text'].split() for char in word } # Character based model
        text_tokens = {}
        text = row["text"]
        text = PUNCTUATION_REGEX.sub("", text) # Remove punctuations
        text = DIGITS_REGEX.sub("", text) # Remove digits
        # Both characters and word based model
        for word in text.split():
            text_tokens[word] = True
            # if not row['label'] in roman_alphabet_languages:
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


def get_data_for_keras(df):
    def _make_tokens(row):
        text = PUNCTUATION_REGEX.sub("", row['text']) # Remove punctuations
        text = DIGITS_REGEX.sub("", text) # Remove digits

        # Both characters and word based model
        roman_alphabet_languages = ['hr','no','sh','nn','id','ceb','gl','ms','vo','la','sl','war',
                                    'pt','en','de','it', 'fr', 'es']
        tokens = ""
        for word in text.split():
            tokens= tokens + word + " "
            if not row['linear_classifier_label'] in roman_alphabet_languages:
                for char in word:
                    tokens= tokens + char + " "

        # if not row['linear_classifier_label'] in roman_alphabet_languages:
        #     text = "".join(text.split()) # Remove all whitespace for trigrams
        trigrams = zip(text, text[1:], text[2:])
        trigrams_list = [tup[0]+tup[1]+tup[2] for tup in trigrams]
        for t in trigrams_list:
            tokens = tokens + t + " "
        return tokens

    data = get_data_for_nltk(df, train=False)
    df['linear_classifier_label'] = linear_classifier.classify_many(data)
    df['tokens'] = df.apply(_make_tokens, axis=1)
    return df


def save_pickle(data, fname):
    with open(fname, "wb") as fname:
        pickle.dump(data, fname)


def neural_network(train_df, cv_df):
    train_df = get_data_for_keras(train_df)
    cv_df = get_data_for_keras(cv_df)
    np.random.seed(7)
    vocab = 50000 # 50000 # 80000
    num_labels = len(train_df['label'].unique())
    tokenizer = text.Tokenizer(num_words=vocab) # character level stuff here
    tokenizer.fit_on_texts(train_df['tokens'])
    train_data = tokenizer.texts_to_matrix(train_df['tokens']) # binary, tf-idf etc here

    print("Saving Tokenizer in a pickle file...")
    save_pickle(tokenizer, "neural_network_tokenizer.pickle")

    print(train_data)
    print(train_data.shape)
    cv_data = tokenizer.texts_to_matrix(cv_df['tokens'])
    encoder = LabelBinarizer()
    encoder.fit(train_df['label'])
    train_data_labels = encoder.transform(train_df['label'])
    print(train_data_labels)
    print(train_data_labels.shape)

    print("Saving Label Encoder in a pickle file...")
    save_pickle(encoder, "neural_network_label_encoder.pickle")

    cv_data_labels = encoder.transform(cv_df['label'])

    model = Sequential()
    model.add(Dense(512, input_shape=(vocab,), activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_data, train_data_labels, batch_size=128,
              epochs=2, verbose=1)
    score = model.evaluate(cv_data, cv_data_labels, batch_size=128, verbose=1)
    print(score[0], score[1])

    print("Saving the model...")
    model.save('neural_network_language.h5')
    print("Model Saved")

    # predictions = []
    # text_labels = encoder.classes_
    # for i in range(len(cv_data_labels)):
    #     prediction = model.predict(np.array([cv_data[i]]))
    #     predicted_label = text_labels[np.argmax(prediction[0])]
    #     predictions.append(predicted_label)

    # 0.8444957032901704 0.7852547027469585 with mode=count 30,000 words and char+trigram+ words as before 2 epochs and 512 one hidden layer
    # 0.789806132737762 0.7953772386353181 with mode=binary rest same as above (train=85%)
    # 0.7697156416061037 0.797139461519464 with mode=binary, 30000, 512+128 two hidden layers and Dropout of 0.3 (train=81.5%)
    # 0.792052224995997 0.7895987870058003 with mode=binary, 50000, 512+256 two hidden layers and Dropout of 0.3, epoch=1 (train=81.5%)
    # 0.7953528065470602 0.79091020861755 with differnt random state of 10
    # 0.8475974188305883 0.797385353152124 with 3 epochs 512+256 and 100,000 words (train=89.74)
    # 0.7625486773634284 0.7986148108903914 with 2 epochs 256+128, 50000 words (train=81.38%)
    # 0.7553023951209269 0.8014015819732198 with 2 epochs 512+256, 50000 words dropout of 0.5 (train=80.3%)
    # 0.747782983069957 0.8051719191054731 with 2 epochs 512 only 50000 words dropout of 0.5 (train=83.4%)
    # 0.744490790118957 0.8050979426938699 with 2 epochs 512. fixed punctuations (train=83.5%)
    # Best so far via labels
    #109803/109803 [==============================] - 372s 3ms/step - loss: 1.0129 - acc: 0.7651
    # 109803/109803 [==============================] - 372s 3ms/step - loss: 0.5069 - acc: 0.8635
    # 12201/12201 [==============================] - 12s 988us/step
    # 0.6223368735481858 0.8280468813054591
    # With linear model
    # 109803/109803 [==============================] - 426s 4ms/step - loss: 1.1211 - acc: 0.7532
    # 109803/109803 [==============================] - 426s 4ms/step - loss: 0.6196 - acc: 0.8421
    # 12201/12201 [==============================] - 12s 983us/step
    # 0.790188166237534 0.802885009405916


def linear_model(train_df, cv_df):
    train_tokenized_data = get_data_for_nltk(train_df)
    cv_tokenized_data = get_data_for_nltk(cv_df)

    print(train_tokenized_data[0], train_tokenized_data[100], train_tokenized_data[10000])
    classifier = SklearnClassifier(SGDClassifier(loss='log', max_iter=5)) # Set up SGD Classifier via scikitlearn wrapper
    print("Training Classifier...")
    classifier.train(train_tokenized_data)
    print("Training Done")

    save_pickle(classifier, "linear_model.pickle")
    cv_accuracy = nltk.classify.accuracy(classifier, cv_tokenized_data)
    print("Cross Validation Accuracy", cv_accuracy)
    print(classification_report(cv_df['label'].tolist(), classifier.classify_many([x for (x,y) in cv_tokenized_data])))

    # print("Plotting Learning Curve...")
    # build_learning_curve(classifier, train_tokenized_data, cv_tokenized_data)


def read_model(cv_df):

    with open("neural_network_tokenizer.pickle", "rb") as fname:
        tokenizer = pickle.load(fname)

    with open("neural_network_label_encoder.pickle", "rb") as fname:
        encoder = pickle.load(fname)

    # print(len(tokenizer.word_index))
    # print(type(tokenizer.word_index))
    # print(tokenizer.word_index.keys())
    # sorted(tokenizer.word_counts, key=tokenizer.word_counts.get, reverse=True)[:1000]
    # import pdb; pdb.set_trace();
    model = load_model('neural_network_language.h5')
    cv_df = get_data_for_keras(cv_df)
    cv_data = tokenizer.texts_to_matrix(cv_df['tokens'])
    # cv_data_labels = encoder.transform(cv_df['label'])

    predictions = model.predict_classes(cv_data)
    labels = encoder.classes_
    predictions_labels = [labels[p] for p in predictions]
    test_labels = cv_df['label'].tolist()
    # print(np.argmax(cv_data_labels, axis=1), predictions, len(predictions))
    # print(cv_df['label'].tolist())
    # print([labels[p] for p in predictions])
    tuples_results = zip(test_labels, predictions_labels)
    df = DataFrame(columns=["text", "tokens", "actual", "predicted"])
    for i, t in enumerate(tuples_results):
        # if t == ('eo', 'zh'):
        #     print(cv_df['text'].iloc[i])
        # if t == ('en', 'zh'):
        #     print(cv_df['text'].iloc[i])
        if t[0]!=t[1]:
            # print(cv_df['text'].iloc[i], t[0], t[1])
            df = df.append({"text":cv_df['text'].iloc[i], "tokens":cv_df['tokens'].iloc[i],
                            "actual": t[0], "predicted": t[1]}, ignore_index=True)

    # print("Saving Analysis")
    # df.to_csv("analysis.csv")

    # print("Saving Confusion Matrix")
    # df = DataFrame(confusion_matrix(test_labels, predictions_labels, labels=labels))
    # df.to_csv('confusion_matrix.csv')
    print(classification_report(test_labels, predictions_labels))


def main():
    data = read_data(text_file, label_file)
    df = DataFrame(data, columns=['text', 'label'])   # Converts list of dict to DataFrame
    df = preprocess(df)
    train_df, cv_df = train_test_split(df, test_size = 0.20, stratify=df['label'], random_state = 7)
    # print(df['label'].value_counts())
    # Split into train and cv sets, stratified so that even languages with lower labels get the same ratio
    # linear_model(train_df, cv_df)
    # neural_network(train_df, cv_df)
    read_model(cv_df)


def scratch():
    data = read_data(text_file, label_file)
    print(data[:15])
    print(len(data[71320]['text'].strip()), data[29124], data[54264], data[107447])
    df = DataFrame(data, columns=['text', 'label'])
    df = df.drop_duplicates(subset='text', keep='first')
    df = df[df['text'].apply(lambda text:len(text.strip())>0)]

    print(df[df['text']=="x i y"])
    # roman_alphabet_languages = ['hr','no','sh','nn','id','ceb','gl','ms','vo','la','sl','war',
    #                             'pt','en','de','it', 'fr', 'es', 'id', 'ms']
    print(df[(df['label']=="de") | (df['label']=="la") |  (df['label']=="sl") | (df['label']=="war")].head(20))
    lookup = {}
    for row in df['text'].tolist():
        for word in row.split():
            lookup[word]=lookup.get(word, 0) + 1

    texts = []
    vals = []
    words_df = DataFrame([])
    for key, val in lookup.items():
        if val > 0:
            texts.append(key)
            vals.append(val)

    print(len(vals))
    # words_df['text'] = texts
    # words_df['count'] = vals
    # words_df = words_df.sort_values('count', ascending=False)
    # print(words_df.head(100))


    # df2 = df['text'].value_counts()
    # g = df.groupby('text')
    # g = g.filter(lambda x:len(x) > 1)
    # print(g.sort_values(by='text'))
    #print(df.loc[2,'text'])
    # print("LENGTH OF UNIQUE SENTENCES", len(df['text'].unique()))
    # print("LENGTH OF SENTENCES", len(df['text']))
    #
    print("LENGTH OF UNIQUE LABELS", len(df['label'].unique()))
    # print("LENGTH OF DATA", len(data))

    # train, test = train_test_split(df, test_size = 0.10, random_state = 7)
    # print(train.shape, test.shape)
    # Find duplicate entries and see if they have different labels
    # Find minimum and maximum length of text
    # Try Logistic Regression Model
    # Learning Curves
    # print(sys.getsizeof(data))


if __name__ == "__main__":
    # scratch()
    main()