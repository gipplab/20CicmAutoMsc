import pickle

# set parameters
classifier_name = 'logistic-regression'
encoding_name = 'tfidf'
sources = ['titles','texts','refs','titles-refs','texts-refs','titles-texts-refs']
source = sources[5]

# load encoder
with open("F:\\zbMath/models/" + source + "_" + encoding_name + "_encoder.pkl","rb") as f:
    encoder = pickle.load(f)
# load classifier
with open("F:\\zbMath/models/" + source + "_" + classifier_name + "_classifier.pkl","rb") as f:
    classifier = pickle.load(f)

# predict msc

def predict_msc(title,text,refs):
    vector = encoder.transform([title + text + refs])
    prediction = classifier.predict(vector)

    return prediction

# test msc prediction

example_idx = 1

with open("F:\\zbMath/data_dicts/data_dict.pkl","rb") as f:
    data_dict = pickle.load(f)

title = data_dict['test'][1][example_idx]
text = data_dict['test'][2][example_idx]
refs = data_dict['test'][3][example_idx]

prediction = predict_msc(title,text,refs)
print(prediction)

print("end")