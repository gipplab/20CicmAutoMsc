import csv
import pickle

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# SET PARAMETERS
nr_digits = 2 # 2 or 5 (e.g., 76 vs. 76M12)
labs_key = 'mscs'

sources = ['titles','texts','refs','titles_text','titles_refs','texts_refs','titles_texts_refs']
src_idxs = {}
src_idxs[labs_key] = 0
src_idxs[sources[0]] = 1
src_idxs[sources[1]] = 2
src_idxs[sources[2]] = 3
src_idxs[sources[3]] = 4
src_idxs[sources[4]] = 5
src_idxs[sources[5]] = 6
src_idxs[sources[6]] = 7

encodings = ['tfidf']
encoding = encodings[0]

classifier_name = "logistic_regression"

# FUNCTION
def parse_file(csv_reader):
    mscs = []
    titles = []
    texts = []
    refs = []
    titles_texts = []
    titles_refs = []
    texts_refs = []
    titles_texts_refs = []
    for row in csv_reader:
        # extract msc
        msc = row[1][:nr_digits]
        mscs.append(msc)
        # extract title
        title = row[2]
        titles.append(title)
        # extract text
        text = row[3]
        texts.append(text)
        # extract refs
        ref = row[4]
        refs.append(ref)
        # unite title and text
        title_text = title + " " + text
        titles_texts.append(title_text)
        # unite title and refs
        title_ref = title + " " + ref
        titles_refs.append(title_ref)
        # unite text and refs
        text_ref = text + " " + ref
        texts_refs.append(text_ref)
        # unite title, text, and refs
        title_text_ref = title + " " + text + " " + ref
        titles_texts_refs.append(title_text_ref)
    return mscs,titles,texts,refs,titles_texts,titles_refs,texts_refs,titles_texts_refs

# OPEN FILES
print("OPEN FILES")

root_path = "F:\\zbMath/"

file_paths = ["train", "test"]

data_dict = {}
for file_path in file_paths:
    with open(root_path + "raw_data/" + file_path + ".csv",'r',encoding='utf8') as f:
        csv_reader = csv.reader(f, delimiter=",")
        # PARSE FILES
        print("PARSE FILES")
        # parse
        print("parse")
        data_dict[file_path] = parse_file(csv_reader)

with open(root_path + "data_dicts/data_dict.pkl",'wb') as f:
    pickle.dump(data_dict,f)

# with open(root_path + "data_dicts/data_dict.pkl",'rb') as f:
#     data_dict = pickle.load(f)

# VECTORIZE
print("VECTORIZE")

# encode
print("encode")

vect_dict = {}
data_idx = 0  # mscs

models = {}

for file_path in file_paths:
    print(file_path)
    vect_dict[file_path] = {}
    # msc binary for y vector
    vect_dict[file_path]['mscs_bin'] = LabelBinarizer().fit_transform(data_dict[file_path][src_idxs[labs_key]])
    # msc label strings for classification report
    vect_dict[file_path]['mscs_str'] = data_dict[file_path][src_idxs[labs_key]]

    # for source in sources:
    # sources[3] = 'titles_text'
    for source in [sources[3]]:
        print(source)

        vect_dict[file_path][source] = {}

        if encoding == encodings[0]:

            if file_path == file_paths[0]: # 'train'
                models[source + " " + encoding] = TfidfVectorizer().fit(data_dict[file_path][src_idxs[source]])
                vect_dict[file_path][source][encoding] = models[source + " " + encoding].transform(data_dict[file_path][src_idxs[source]])
            if file_path == file_paths[1]: # 'test'
                vect_dict[file_path][source][encoding] = models[source + " " + encoding].transform(data_dict[file_path][src_idxs[source]])

# # save vectorizer model
# print("save vectorizer")
# # sources[6]: 'titles_texts_refs'
# with open(root_path + sources[6] + "-" + encoding + ".pkl",'wb') as f:
#     pickle.dump(models[sources[6] + " " + encoding],f)

del data_dict

with open(root_path + "data_dicts/vect_dict.pkl",'wb') as f:
    pickle.dump(vect_dict,f)

# with open(root_path + "data_dicts/vect_dict.pkl",'rb') as f:
#     vect_dict = pickle.load(f)

# CLASSIFY
print("CLASSIFY")

eval_dict = {}
#sources = [sources[4]]
for source in sources:
    print(source)
    eval_dict[source] = {}

    X_train, y_train = vect_dict['train'][source][encoding],vect_dict['train']['mscs_str']
    X_test, y_test = vect_dict['test'][source][encoding],vect_dict['test']['mscs_str']

    # fit classifier
    print("fit classifier")
    classifier = LogisticRegression(verbose=0).fit(X_train,y_train)
    #classifier = SVC(verbose=0).fit(X_train,y_train)
    #classifier = MLPClassifier(verbose=1,max_iter=25).fit(X_train,y_train) #hidden_layer_sizes=(500,)

    # # save classifier model
    # print("save classifier")
    # # sources[6]: 'titles_texts_refs'
    # if source == sources[6]:
    #     with open(root_path + classifier_name + "-classifier.pkl", 'wb') as f:
    #         pickle.dump(classifier, f)

    # save evaluation results
    print("save evalution")
    with open(root_path + "results/" + classifier_name + ".pkl", 'wb') as f:
        pickle.dump(eval_dict, f)

    # evaluate classifier
    print("evaluate classifier")
    y_pred = classifier.predict(X_test)
    #y_pred = LabelBinarizer().fit_transform(classifier.predict(X_test))
    y_true = y_test
    #y_true = LabelBinarizer().fit_transform(y_test)

    eval_dict[source][encoding] = {}
    eval_dict[source][encoding]['pred'] = y_pred
    eval_dict[source][encoding]['conf'] = classifier.predict_proba(X_test) # confidence

    eval_dict[source][encoding]['score'] = classifier.score(X_test,y_test)
    eval_dict[source][encoding]['accuracy'] = metrics.accuracy_score(y_true=y_true,y_pred=y_pred)

    eval_dict[source][encoding]['precision'] = metrics.precision_score(y_true=y_true,y_pred=y_pred,average='weighted')
    eval_dict[source][encoding]['recall'] = metrics.recall_score(y_true=y_true,y_pred=y_pred,average='weighted')
    eval_dict[source][encoding]['f1'] = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

    #probas_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #eval_dict[source][encoding]['precision_recall_curve'] = metrics.precision_recall_curve(y_true=y_true,probas_pred=probas_pred)

    eval_dict[source][encoding]['classification_report'] = metrics.classification_report(y_true=y_true, y_pred=y_pred)#,target_names=y_test)

#########
# RESULTS
#########

# save evaluation results
print("save evalution")
with open(root_path + "results/" + classifier_name + ".pkl",'wb') as f:
    pickle.dump(eval_dict,f)

print("end")