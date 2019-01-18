from pandas import DataFrame
from keras.models import load_model
import pickle
import re
import json

# For text preprocessing of test data
from build_model import get_data_for_keras

test_data_file = "./languages_homework_X_test.json.txt"
test_predictions_file = "predictions.json.txt"
PUNCTUATION_REGEX = re.compile(r'([!"#$%&()\'*+,-.:;\/<=>?@[\]^_`{|}~\\])')
DIGITS_REGEX = re.compile(r'\d')


def read_data(fname):
    data = []
    with open(fname, 'r') as test_data:
        for line in test_data:
            object = json.loads(line)
            data.append(object)
    return data


def make_predictions(df):
    print("Loading the Tokenizer...")
    with open("neural_network_tokenizer.pickle", "rb") as fname:
        tokenizer = pickle.load(fname)

    with open("neural_network_label_encoder.pickle", "rb") as fname:
        encoder = pickle.load(fname)

    print("Loading Neural Network Model...")
    model = load_model('neural_network_language.h5')
    print("Preparing the Data...")
    df = get_data_for_keras(df)
    test_data = tokenizer.texts_to_matrix(df['tokens'])

    predictions = model.predict_classes(test_data)
    labels = encoder.classes_
    predictions = [labels[p] for p in predictions]

    return predictions


def save_predictions(predictions):
    print("Saving the File")
    with open(test_predictions_file, 'w') as fname:
        for prediction in predictions:
            json.dump({"classification": str(prediction)}, fname)
            fname.write("\n")


def main():
    data = read_data(test_data_file)
    df = DataFrame(data, columns=['text'])  # Converts list of dict to DataFrame
    predictions = make_predictions(df)
    save_predictions(predictions)

if __name__ == "__main__":
    main()