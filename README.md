# language-identifier-nlp
Identifying different languages from their scripts using NLP techniques and Deep Learning

Process that was used to select the best model:
1) Data Understanding: First step was to understand the data. There are around 122k training examples, and 56 different language labels.
   This is a problem of multi-class text classification. I wanted to start with a linear model first and build from there.
2) Data Preparation: At the onset, I realized that training on words alone is not going to be sufficient, as it is possible that words seen in training data are
   not repeated later. It is more important to understand inherent "structure" of each language. I started using unigrams and trigrams.
   n-grams are particularly useful for languages like Chinese and Japanese, in which characters themselves can be words.
   I used stratified train and test sets to make sure each class got the right ratio in the training and testing sets (as some classes
   were skewed- this can be seen in the classification reports below with `ceb` having 29 examples but `en` having 446 examples.
3) Linear Model: I used Logistic Regression Classifier with Stochastic Gradient Descent (to speed up the learning process).
   To evaluate the model, I used learning curve to understand bias and variance (with a cross validation data set). This was important esp. to decide if I wanted to use
   neural network at all or not. Based on learning curve analysis, it was clear that linear model had high bias as well as some variance.
   Accuracy alone is not sufficient for skewed data set, weighted F1-score of 0.79 was achieved via linear model.
4) Neural Network: I used a basic 2-layer neural network with first layer of neurons (and relu activation) followed by a linear layer.
   I compared training and test accuracy to understand bias and variance. With something akin to grid search, I zeroed on 50,000 words and
   512 neurons for the first layer. This iterative process helped me choose a Dropout of 0.5 and 2 epochs.
5) Model Evaluation: I used confusion matrix and classification reports to understand precision, recall and F1-scores. I also saw which examples
   were being consistently misclassified by the neural network. It seems like some examples just have digits or punctuations - and are really
   hard to classify. This process also enabled me to realize that roman alphabet languages do not perform well in general - and one reason could
   be that they all share characters.
6) Linear Model followed by Neural Network: This made me go to using Linear Model to first guess if the text is possibly in one of the roman alphabet languages or not.
   By low performing, I mean the ones with low F1-scores. This information is then used by the preprocessing stage to use character level tokens or not.
   This process enables the F1-score to reach between 0.80 and 0.81.
7) Saving the Models to Predict: The neural network has been saved in a h5 file, and label encoder and tokenizer in pickle files. Linear Model
   is also saved in pickle file.

Confusion Matrix stored in confusion_matrix.csv and misclassifications analysis in misclassifications_analysis.csv
Linear Model Expected Performance Below:
```
             precision    recall  f1-score   support

         ar       0.97      0.94      0.96       815
         az       0.93      0.81      0.87       373
         be       0.91      0.80      0.85       392
         bg       0.84      0.86      0.85       542
         ca       0.63      0.73      0.68       591
         ce       0.81      0.67      0.73       106
        ceb       0.80      0.49      0.61        57
         cs       0.89      0.76      0.82       456
         da       0.65      0.73      0.69       577
         de       0.83      0.82      0.83       915
         el       0.96      0.88      0.92       541
         en       0.57      0.85      0.68       892
         eo       0.86      0.64      0.73       386
         es       0.68      0.86      0.76      1020
         et       0.92      0.79      0.85       179
         eu       0.73      0.86      0.79       342
         fa       0.96      0.88      0.92       387
         fi       0.91      0.83      0.87       310
         fr       0.81      0.81      0.81       767
         gl       0.71      0.55      0.62       418
         he       1.00      0.97      0.98       456
         hi       0.96      0.88      0.92       410
         hr       0.51      0.44      0.47       360
         hu       0.92      0.88      0.90       539
         hy       0.99      0.90      0.94       442
         id       0.61      0.65      0.63       323
         it       0.74      0.84      0.79       949
         ja       0.94      0.84      0.89       337
         ka       1.00      0.98      0.99       226
         kk       0.93      0.71      0.80       218
         ko       1.00      0.94      0.97       244
         la       0.73      0.55      0.63       244
      lorem       0.93      1.00      0.96       660
         lt       0.95      0.79      0.86       308
         ms       0.64      0.49      0.56       240
         nl       0.89      0.77      0.83       486
         nn       0.73      0.45      0.55       272
         no       0.57      0.58      0.57       458
         pl       0.78      0.86      0.82       402
         pt       0.71      0.64      0.67       529
         ro       0.84      0.77      0.81       581
         ru       0.80      0.80      0.80       787
         sh       0.57      0.55      0.56       514
         sk       0.84      0.64      0.73       211
         sl       0.75      0.63      0.69       306
         sr       0.86      0.84      0.85       401
         sv       0.80      0.69      0.74       504
         th       1.00      0.86      0.92       281
         tr       0.85      0.79      0.82       490
         uk       0.86      0.85      0.85       669
         ur       1.00      0.88      0.94       181
         uz       0.91      0.75      0.82       171
         vi       0.85      0.89      0.87       634
         vo       0.94      0.46      0.62        71
        war       0.78      0.69      0.73       101
         zh       0.41      0.90      0.57       330
         
avg / total       0.81      0.79      0.79     24401
```
```
Neural Network Expected Performance Below:
             precision    recall  f1-score   support

         ar       0.99      0.96      0.98       815
         az       0.98      0.91      0.95       373
         be       0.97      0.95      0.96       392
         bg       0.97      0.94      0.96       542
         ca       0.81      0.87      0.84       591
         ce       0.98      0.85      0.91       106
        ceb       0.84      0.82      0.83        57
         cs       0.96      0.92      0.94       456
         da       0.89      0.84      0.86       577
         de       0.84      0.93      0.88       915
         el       0.99      0.93      0.96       541
         en       0.59      0.94      0.72       892
         eo       0.93      0.82      0.87       386
         es       0.87      0.91      0.89      1020
         et       0.96      0.93      0.95       179
         eu       0.91      0.92      0.92       342
         fa       0.99      0.91      0.95       387
         fi       0.97      0.90      0.93       310
         fr       0.84      0.93      0.89       767
         gl       0.85      0.83      0.84       418
         he       1.00      0.98      0.99       456
         hi       0.99      0.91      0.95       410
         hr       0.79      0.75      0.77       360
         hu       0.98      0.95      0.96       539
         hy       0.99      0.93      0.96       442
         id       0.85      0.83      0.84       323
         it       0.88      0.91      0.90       949
         ja       0.99      0.92      0.95       337
         ka       1.00      0.99      0.99       226
         kk       0.98      0.88      0.92       218
         ko       1.00      0.99      0.99       244
         la       0.91      0.79      0.84       244
      lorem       1.00      1.00      1.00       660
         lt       0.97      0.92      0.95       308
         ms       0.85      0.76      0.80       240
         nl       0.93      0.93      0.93       486
         nn       0.88      0.76      0.82       272
         no       0.73      0.83      0.78       458
         pl       0.86      0.93      0.89       402
         pt       0.90      0.83      0.86       529
         ro       0.92      0.86      0.89       581
         ru       0.91      0.92      0.91       787
         sh       0.84      0.78      0.81       514
         sk       0.95      0.90      0.92       211
         sl       0.95      0.83      0.88       306
         sr       0.96      0.95      0.95       401
         sv       0.91      0.84      0.87       504
         th       0.98      0.90      0.94       281
         tr       0.95      0.89      0.92       490
         uk       0.98      0.92      0.95       669
         ur       0.99      0.98      0.98       181
         uz       0.96      0.89      0.92       171
         vi       0.98      0.94      0.96       634
         vo       0.93      0.76      0.84        71
        war       0.91      0.77      0.83       101
         zh       0.97      0.91      0.94       330

avg / total       0.91      0.90      0.90     24401
```

How I engineered the features?
1) n-grams: At the onset, I realized that training on words alone is not going to be sufficient, as it is possible that words seen in training data are
   not repeated later. It is more important to understand inherent "structure" of each language. I started using unigrams and trigrams.
   n-grams are particularly useful for languages like Chinese and Japanese, in which characters themselves can be words.
   I used stratified train and test sets to make sure each class got the right ratio in the training and testing sets (as some classes
   were skewed- this can be seen in the classification reports below with `ceb` having 29 examples but `en` having 446 examples.
2) Unique nature of Roman Alphabet languages: I used confusion matrix and classification reports to understand precision, recall and F1-scores. I also saw which examples
   were being consistently misclassified by the neural network. It seems like some examples just have digits or punctuations - and are really
   hard to classify. This process also enabled me to realize that roman alphabet languages do not perform well in general - and one reason could
   be that they all share characters.
3) This made me go to using Linear Model to first guess if the text is possibly in one of the roman alphabet languages or not.
   This information is then used by the preprocessing stage to use character level tokens or not.
   In summary,
   Roman Alphabet Languages use words and trigrams.
   All other languages use words, unigrams and trigrams.
   This information is derived from linear model first predicting whether it thinks its a roman alphabet language or not.

How I chose the best model? (also in performance.txt)
1) Data Understanding: First step was to understand the data. There are around 122k training examples, and 56 different language labels.
   This is a problem of multi-class text classification. I wanted to start with a linear model first and build from there.
2) Data Preparation: At the onset, I realized that training on words alone is not going to be sufficient, as it is possible that words seen in training data are
   not repeated later. It is more important to understand inherent "structure" of each language. I started using unigrams and trigrams.
   n-grams are particularly useful for languages like Chinese and Japanese, in which characters themselves can be words.
   I used stratified train and test sets to make sure each class got the right ratio in the training and testing sets (as some classes
   were skewed- this can be seen in the classification reports below with `ceb` having 29 examples but `en` having 446 examples.
3) Linear Model: I used Logistic Regression Classifier with Stochastic Gradient Descent (to speed up the learning process).
   To evaluate the model, I used learning curve to understand bias and variance (with a cross validation data set). This was important esp. to decide if I wanted to use
   neural network at all or not. Based on learning curve analysis, it was clear that linear model had high bias as well as some variance.
   Accuracy alone is not sufficient for skewed data set, weighted F1-score of 0.79 was achieved via linear model.
4) Neural Network: I used a basic 2-layer neural network with first layer of neurons (and relu activation) followed by a linear layer.
   I compared training and test accuracy to understand bias and variance. With something akin to grid search, I zeroed on 50,000 words and
   512 neurons for the first layer. This iterative process helped me choose a Dropout of 0.5 and 2 epochs.
5) Model Evaluation: I used confusion matrix and classification reports to understand precision, recall and F1-scores. I also saw which examples
   were being consistently misclassified by the neural network. It seems like some examples just have digits or punctuations - and are really
   hard to classify. This process also enabled me to realize that roman alphabet languages do not perform well in general - and one reason could
   be that they all share characters.
6) Linear Model followed by Neural Network: This made me go to using Linear Model to first guess if the text is possibly in one of the roman alphabet languages or not.
   By low performing, I mean the ones with low F1-scores. This information is then used by the preprocessing stage to use character level tokens or not.
   This process enables the F1-score to reach between 0.80 and 0.81.
7) Saving the Models to Predict: The neural network has been saved in a h5 file, and label encoder and tokenizer in pickle files. Linear Model
   is also saved in pickle file.

What would I have done if I had extra time?
1) Evaluated using tree based model up-front to separate the languages into Roman script or Not Roman Script.
   I used linear model here which is also classifying into one of the 56 languages and whose input is then used to feature engineer for neural network.
   It will be better to have binary classifier here instead.
2) I used Keras to build the neural network here, with more time I would have written in TensorFlow to optimize at the low-level even more.
3) With more time, I would write a proper script to do grid search for hyper-parameters.


Software Dependencies: Details of these in requirements.txt (including version numbers).
But overall main libraries for machine learning: numpy, pandas, keras (with tensorflow backend), seaborn, scikit-learn and nltk

Model Binary: `neural_network_language.h5` is the main neural network. neural_network_label_encoder.pickle and neural_network_tokenizer.pickle are
label encoder and tokenizer respectively for data preparation. `linear_model.pickle` is serialized linear model which is used by the data preparation
stage of neural network.

Overall, it was fun to play around with the language dataset!
