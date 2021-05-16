# libraries
from keras import callbacks
import random
from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.models import Sequential
import numpy as np
import pickle
import json
import os
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# download punkt and wordnet
# punkt is the pre-trained tokenizer we will use for the Enligsh language

# we will be ignoring ? and ! because they are redundant as we are not yet interpreting intented tone (who do you think I am. Mr Musk!? Hell no!)


# init
words = []
classes = []
documents = []
ignore_words = []
dirname = os.path.dirname(__file__)
data_filename = os.path.join(dirname, 'intents.json')  # quick explanation - my directories are screwed so i hada to specify double the trouble
data_file = open(data_filename).read()
intents = json.loads(data_file)

# tokenizing time

for intent in intents["intents"]:
    for pattern in intent["patterns"]:

        # takes each of the words and tokenizes it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent["tag"]))

        # adding classes to our class list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# lemmatimzing time
# dump in pickle file

words = [lemmatizer.lemmatize(w.lower())
         for w in words if w not in ignore_words]

words = sorted(list(set(words)))


classes = sorted(list(set(classes)))

print(len(documents), "documents")

print(len(classes), "classes")

print(len(words), "lemmatized unique words")
dirname = os.path.dirname(__file__)
filenameobj1 = os.path.join(dirname, 'words.pkl')
pickle.dump(words, open(filenameobj1, "wb"))
dirname = os.path.dirname(__file__)
filenameobj2 = os.path.join(dirname, 'classes.pkl')
pickle.dump(classes, open(filenameobj2, "wb"))

# time to initialize model training

training = []
output_empty = [0] * len(classes)
for doc in documents:
    # init bag
    bag = []
    # list the tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(
        word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
model.summary()

# i'll be honest I am still a bit confused on this part.
# I am using a tutorial though.

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy",
              optimizer=sgd, metrics=["accuracy"])

# OPTIONAL
# Avoiding undergitting

earlystopping = callbacks.EarlyStopping(
    monitor="loss", mode="min", patience=5, restore_best_weights=True)
callbacks = [earlystopping]

# now for flask
# i don't know flask
# like the bottle?

hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)
dirname = os.path.dirname(__file__)
filenameobj = os.path.join(dirname, 'chatbot_model.h5')
model.save(filenameobj, hist)
print("Model Created!")
