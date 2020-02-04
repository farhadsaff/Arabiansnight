# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 08:32:13 2020

@author: fsaff
"""

import spacy 
import numpy as np
import pandas as pd

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
sp = spacy.load('en_core_web_sm')
nlp = spacy.load('en')

file_content = open('Arabian-NightsA.txt', encoding="utf8").read()
file_content=file_content.lower()
file_content1=file_content.split()
f=[]
f=list(file_content1)
f2=pd.DataFrame(columns={"A"})
f2["A"]=f

f2['A'] = f2['A'].str.replace('\d+', '')
#Preprocessing of a text and filtering 
f2=(f2[f2['A'].apply(lambda x: len(x.split('-')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split(',')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split(':')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split('"')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split('?')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split('.')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split('*')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split('.')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split(';')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split('%')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split('‘')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split('“')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split('9')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split('!')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split('(')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split(')')) < 2)])
f2=(f2[f2['A'].apply(lambda x: len(x.split('=')) < 2)])
lf2=list(f2["A"])

f3=str(f2)

parser = English()

document = nlp(f3)

lemmas = [token.lemma_ for token in document if not token.is_stop]

import spacy 
nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])
from nltk.corpus import stopwords
#updating stopwords
stops = stopwords.words("english")
nlp.Defaults.stop_words |= {"",",","'","@","$",":","-",".","AHMAGHHHHHHHHHHH","(",")"}

def normalize(comment, lowercase, remove_stopwords):
    if lowercase:
        comment = comment.lower()
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in stops):
                lemmatized.append(lemma)
    return " ".join(lemmatized)



f=pd.DataFrame()
f["A"]=f2["A"]
f['A']= f['A'].apply(normalize, lowercase=True, remove_stopwords=True)
f['A'].replace('', np.nan, inplace=True)
f.dropna(subset=['A'], inplace=True)


LLL=[]
LLL=list(f["A"])
writer = pd.ExcelWriter('BestA11.xlsx', engine='xlsxwriter')
writer0 = pd.ExcelWriter('BestA12.xlsx', engine='xlsxwriter')
writer2 = pd.ExcelWriter('BestA13.xlsx', engine='xlsxwriter')

NNNN=200
new_lst=list(range(0,NNNN))
import gensim 
from gensim.models import Word2Vec 
import random
S1=[]
S2=[]
S3=[]
S4=[]
for iii in range (1,NNNN):

# Training the neural language network in a loop using Skip-gram method
    NN=random.randint(1,101)
    model3 = gensim.models.Word2Vec([LLL],min_count = 1, 
					 seed=NN,sg=0, size = 150, window = 5)
    model3.build_vocab([LLL], update=True)

# Finding the similarities between the-most-common-words and "Persian", "Arab", "Persins", and "Arabs".    
    S1.append(model3.similarity("persian","man")) 
    S2.append(model3.similarity("arab","man"))     
    S3.append(model3.similarity("persians","man")) 
    S4.append(model3.similarity("arabic","man"))  
   

    S1.append(model3.similarity("persian","king")) 
    S2.append(model3.similarity("arab","king"))
    S3.append(model3.similarity("persians","king")) 
    S4.append(model3.similarity("arabs","king"))
   
 

    S1.append(model3.similarity("persian","god")) 
    S2.append(model3.similarity("arab","god")) 
    

    S1.append(model3.similarity("persian","girl")) 
    S2.append(model3.similarity("arab","girl")) 
    


    S1.append(model3.similarity("persian","morning")) 
    S2.append(model3.similarity("arab","morning"))
    

    S1.append(model3.similarity("persian","night")) 
    S2.append(model3.similarity("arab","night"))
    


    S1.append(model3.similarity("persian","love")) 
    S2.append(model3.similarity("arab","love"))  
    

    S1.append(model3.similarity("persian","old")) 
    S2.append(model3.similarity("arab","old"))  
    


    S1.append(model3.similarity("persian","fortunate")) 
    S2.append(model3.similarity("arab","fortunate")) 
    

    S1.append(model3.similarity("persian","slave")) 
    S2.append(model3.similarity("arab","slave"))  
   
 

    S1.append(model3.similarity("persian","vizier")) 
    S2.append(model3.similarity("arab","vizier"))  
   

   
    S1.append(model3.similarity("persian","father")) 
    S2.append(model3.similarity("arab","father")) 
    


    S1.append(model3.similarity("persian","young")) 
    S2.append(model3.similarity("arab","young"))  
    


    S1.append(model3.similarity("persian","heart")) 
    S2.append(model3.similarity("arab","heart"))
    


    S1.append(model3.similarity("persian","kiss")) 
    S2.append(model3.similarity("arab","kiss"))  
   
    S3.append(model3.similarity("persians","god")) 
    S4.append(model3.similarity("arabs","god"))  


    S3.append(model3.similarity("persians","girl")) 
    S4.append(model3.similarity("arabs","girl"))  


    S3.append(model3.similarity("persians","morning")) 
    S4.append(model3.similarity("arabs","morning"))  


    S3.append(model3.similarity("persians","night")) 
    S4.append(model3.similarity("arabs","night"))  
 

    S3.append(model3.similarity("persians","love")) 
    S4.append(model3.similarity("arabs","love"))  


    S3.append(model3.similarity("persians","old")) 
    S4.append(model3.similarity("arabs","old"))  
 


    S3.append(model3.similarity("persians","fortunate")) 
    S4.append(model3.similarity("arabs","fortunate"))  


    S3.append(model3.similarity("persians","slave")) 
    S4.append(model3.similarity("arabs","slave"))  


    S3.append(model3.similarity("persians","vizier")) 
    S4.append(model3.similarity("arabs","vizier"))  
 
   
    S3.append(model3.similarity("persians","father")) 
    S4.append(model3.similarity("arabs","father"))  
 

    S3.append(model3.similarity("persians","young")) 
    S4.append(model3.similarity("arabs","young"))  


    S3.append(model3.similarity("persians","heart")) 
    S4.append(model3.similarity("arabs","heart"))  


    S3.append(model3.similarity("persians","kiss")) 
    S4.append(model3.similarity("arabs","kiss")) 


# Postprocessing and structuring the data for visualization and statistical analysis
NS1=np.array([S1,S3])
MNS1=np.array([])
MNS1=np.average(NS1, axis=0)
LL1n=MNS1.reshape(15,NNNN-1)
LL1=np.average(LL1n, axis=1)
SLL1=np.std(LL1n, axis=1)
DMNS1=pd.DataFrame(LL1)

NS2=np.array([S2,S4])
MNS2=np.array([])
MNS2=np.average(NS2, axis=0)
LL2n=MNS2.reshape(15,NNNN-1)
LL2=np.average(LL2n, axis=1)
SLL2=np.std(LL2n, axis=1)
DMNS2=pd.DataFrame(LL2)

MNS1=np.average(MNS1, axis=0)
SS=pd.DataFrame(S1)
SS1=pd.DataFrame(S2)
SS2=pd.DataFrame(S3)
      
DMNS1.to_excel(writer,index = False,  sheet_name= 'run%d' %(iii)) 
DMNS2.to_excel(writer0,index = False,  sheet_name= 'run%d' %(iii)) 




writer.save()
writer0.save()



import spacy
from collections import Counter
from collections import Counter
word_freq = Counter(LLL)
common_words = word_freq.most_common(100)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.clf()

labels = ['man', 'king', 'god', 'girl', 'morning']
persian_means = [LL1[0], LL1[1], LL1[2], LL1[3], LL1[4]]
Arab_means = [LL2[0], LL2[1], LL2[2], LL2[3], LL2[4]]
error1=[SLL1[0],SLL1[1],SLL1[2],SLL1[3],SLL1[4]]
        
error2=[SLL2[0],SLL2[1],SLL2[2],SLL2[3],SLL2[4]]
x = np.arange(len(labels))  
width = 0.35  
ind = np.arange(5)    
width = 0.35      


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, persian_means, width, yerr=error1,label='Persianness')
rects2 = ax.bar(x + width/2, Arab_means, width,yerr=error2, label='Arabness')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Similarity')
#ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


#plt.legend((p1[0], p2[0]), ('Men', 'Women'))



fig.tight_layout()

plt.show()


plt.clf()

labels = ['night', 'love', 'old', 'fortunate', 'slave']
persian_means = [LL1[5], LL1[6], LL1[7], LL1[8], LL1[9]]
Arab_means = [LL2[5], LL2[6], LL2[7], LL2[8], LL2[9]]
error3=[SLL1[5],SLL1[6],SLL1[7],SLL1[8],SLL1[9]]
        
error4=[SLL2[5],SLL2[6],SLL2[7],SLL2[8],SLL2[9]]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, persian_means, width,yerr=error3, label='Persianness')
rects2 = ax.bar(x + width/2, Arab_means, width,yerr=error4, label='Arabness')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Similarity')
#ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()







fig.tight_layout()

plt.show()

plt.clf()

labels = ['vizier', 'father', 'young', 'heart', 'kiss']
persian_means = [LL1[10], LL1[11], LL1[12], LL1[13], LL1[14]]
Arab_means = [LL2[10], LL2[11], LL2[12], LL2[13], LL2[14]]
error5=[SLL1[10], SLL1[11], SLL1[12], SLL1[13], SLL1[14]]
        
error6=[SLL2[10], SLL2[11], SLL2[12], SLL2[13], SLL2[14]]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, persian_means, width, yerr=error5,label='Persianness')
rects2 = ax.bar(x + width/2, Arab_means, width,yerr=error6, label='Arabness')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Similarity')
#ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()







fig.tight_layout()

plt.show()
