from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from annoy import AnnoyIndex

import os
import numpy as np
import pickle

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
feature_file = 'data/training_features_augmented.pickle'

with open(f'{base_path}/{feature_file}', 'rb') as fp:
    data = pickle.load(fp)

X = []
y = []

for key, value in data.items():
    for item in value:
        X.append(item)
        y.append(key)

X = np.asanyarray(X)
y = np.asanyarray(y)

le = preprocessing.LabelEncoder()
le.classes_ = np.load('labels.npy')
y_label = le.transform(y)
#
# # y_label = le.fit(y)
# # np.save('labels.npy', y_label.classes_)


X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.2, random_state=0)


def train():
    ann_tree = AnnoyIndex(512)
    for embed, label in zip(X, y_label):
        ann_tree.add_item(label, embed)
    ann_tree.build(10)
    ann_tree.save('speaker_emb.bin')


def test():
    ground_truth = []
    predicted = []
    ann_tree = AnnoyIndex(512)
    ann_tree.load('speaker_emb.bin')
    for embed, label in zip(X_train, y_train):
        pred_label = ann_tree.get_nns_by_vector(embed, 1)
        ground_truth.append(label)
        predicted.append(pred_label)
    print(accuracy_score(ground_truth, predicted))

