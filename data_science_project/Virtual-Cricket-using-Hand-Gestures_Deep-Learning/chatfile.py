import json
import nltk
from nltk.stem import PorterStemmer
import numpy as np
import math
from collections import Counter
import random
from django.shortcuts import render

from django.db.models.query import QuerySet
def chat(request):
    query = request.GET.get('query')
    inp=query 
    with open("C:/Users/Shivam/Desktop/dl_project/cricket/Virtual_Cricket_project_Aadesh/Virtual_Cricket_project/Virtual_Cricket/Virtual_Cricket/ChatBot.json") as f:
        data = json.load(f)
    words=[]
    docs_y=[]
    labels=[]
    ps=PorterStemmer()
    for intent in data["intents"]:
        for pattern in intent["patterns"]:        
            w=nltk.word_tokenize(pattern)
            w=[ps.stem(i.lower()) for i in w if i != "?"]
            words.append(w)
            docs_y.append(intent["tag"])
            if intent["tag"] not in labels:
                labels.append(intent["tag"])
    result_index,value=similarity(inp,words)
    tag=docs_y[result_index]
    if value> 0.5:
        for t in data["intents"]:
            if t["tag"]==tag:
                responses=t['responses']
                a=(random.choice(responses))
    else:
        a= "Didn't understand"
    ans=a
    return render(request, 'VideoStreamerApp/home.html', {'ans': ans, 'query': query})
def text_to_vector(text):
    words=nltk.word_tokenize(text)
    return Counter(words) #returns dict with letter as key and occurance as value
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys()) # |A ins B|
    numerator = sum([vec1[x] * vec2[x] for x in intersection]) # |A ins B|

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())]) # |A|
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())]) # |B|
    denominator = math.sqrt(sum1) * math.sqrt(sum2) #squqre root of |A|.|B|
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator
def bags_of_words(s,words):
    bags=[0 for _ in range(len(words))]
    s_words=nltk.word_tokenize(s)
    s_words=[ps.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bags[i]=1
    return np.array(bags)
def similarity(inp,words):
    sim=[]
    vector2 = text_to_vector(inp)
    for i in range(len(words)):
        s=""
        s=' '.join(map(str, words[i]))
        vector1=text_to_vector(s)
        cosine = get_cosine(vector1, vector2)
        sim.append(cosine)
    return np.argmax(sim),np.max(sim)
