import nltk
import string
from nltk.corpus import wordnet as wn
from nltk.corpus import genesis
#nltk.download('genesis')
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')
genesis_ic = wn.ic(genesis, False, 0.0)

from itertools import izip_longest

#import numpy as np
#import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import roc_auc_score

input_file1 = open("apple.txt", "r")
input_file2 = open("bra.txt", "r")
input_file3 = open("fruit.txt", "r")
input_file4 = open("rock.txt", "r")

text1 = input_file1.read()
text2 = input_file2.read()
text3 = input_file3.read()
text4 = input_file4.read()

punctuation = string.maketrans("!.,()", 5*" ")

text1 = text1.translate(punctuation)
text2 = text2.translate(punctuation)
text3 = text3.translate(punctuation)
text4 = text4.translate(punctuation)

text1 = (text1.lower())
text2 = (text2.lower())
text3 = (text3.lower())
text4 = (text4.lower())

stop_words = set(stopwords.words('english')) 

tokens1 = word_tokenize(text1)
tokens2 = word_tokenize(text2)
tokens3 = word_tokenize(text3)
tokens4 = word_tokenize(text4)

filtered_sentence1 = [w for w in tokens1 if not w in stop_words] 
filtered_sentence1 = [] 
for w in tokens1: 
    if w not in stop_words: 
        filtered_sentence1.append(w) 
        
filtered_sentence2 = [w for w in tokens2 if not w in stop_words] 
filtered_sentence2 = [] 
for w in tokens2: 
    if w not in stop_words: 
        filtered_sentence2.append(w) 

filtered_sentence3 = [w for w in tokens3 if not w in stop_words] 
filtered_sentence3 = [] 
for w in tokens3: 
    if w not in stop_words: 
        filtered_sentence3.append(w) 

filtered_sentence4 = [w for w in tokens4 if not w in stop_words] 
filtered_sentence4 = [] 
for w in tokens4: 
    if w not in stop_words: 
        filtered_sentence4.append(w) 
#print(tokens) 
#print(filtered_sentence) 

score2 = 0
score3 = 0
score4 = 0

for i in filtered_sentence1:
    if i in filtered_sentence2:
        score2 = score2 + 1

for i in filtered_sentence1:
    if i in filtered_sentence3:
        score3 = score3 + 1

for i in filtered_sentence1:
    if i in filtered_sentence4:
        score4 = score4 + 1

if score2 == max(score2, score3, score4):
    print("bra")

if score3 == max(score2, score3, score4):
    print("fruit")

if score4 == max(score2, score3, score4):
    print("rock")

print "Ranking: ", max(score2, score3, score4)

#X_train = [filtered_sentence2, filtered_sentence3, filtered_sentence4]
#y_train = filtered_sentence1
#classifier = KNeighborsClassifier(n_neighbors=1)
#classifier.fit(X_train, y_train)

#y_pred_final = classifier.predict(y_train)

#print(y_pred_final)

