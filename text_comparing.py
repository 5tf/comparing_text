#import nltk
import string
#from nltk.corpus import wordnet as wn
#from nltk.corpus import genesis
#nltk.download('genesis')
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')
#genesis_ic = wn.ic(genesis, False, 0.0)

#from itertools import izip_longest

from nltk.tokenize import word_tokenize
#from nltk.stem.porter import PorterStemmer
#from nltk.stem import SnowballStemmer
#from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords

input_file1 = open("text1.txt", "r")
input_file2 = open("text2.txt", "r")

text1 = input_file1.read()
text2 = input_file2.read()

punctuation = string.maketrans("!.,()", 5*" ")

text1 = text1.translate(punctuation)
text2 = text2.translate(punctuation)

text1 = (text1.lower())
text2 = (text2.lower())

stop_words = set(stopwords.words('english')) 

tokens1 = word_tokenize(text1)
tokens2 = word_tokenize(text2)

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

score = 0

for i in filtered_sentence1:
    if i in filtered_sentence2:
        score = score + 1

new_file = open("ranking.txt", "w+")
new_file.write(str(score))
