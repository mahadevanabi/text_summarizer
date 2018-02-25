import kivy
import codecs
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.properties import StringProperty
from kivy.uix.scrollview import ScrollView
from kivy.factory import Factory
import networkx as nx
import nltk
import numpy
from scipy import sparse
import math
import re


Window.size = (800,500)
Builder.load_file('myui.ky')

class Myui(TabbedPanel):
    finaltxt = StringProperty(str(""))
    tfidf = {}
    mytf={}
    myidf={}
    stopwords = []
    sentence_tokenizer=[]
    stopword_file = open("stopwords.txt", "r")
    stopwords = [line.strip() for line in stopword_file]
    limit=0;

    def step(self,a):
        f=codecs.open("input_file.txt",mode='w+',encoding='utf-8')
        f.write(a)
        f.close()
        self.textrank("input_file.txt")
        with open("summari.txt") as f:
            content=f.read()
            self.finaltxt = str(content)

    def get_tokens(self,str):
        return re.findall(r"<a.*?/a>|<[^\>]*>|[\w'@#]+", str.lower())
    
    def get_idf(self,term):
        count=0  
        setter=set()
        if term in self.stopwords:
            return 0
        setter.add(term)
        
        for i in range(len(self.sentence_tokenizer)):
            tokens_idf = self.get_tokens(self.sentence_tokenizer[i])            
            tokens_set_idf = set(tokens_idf)
            
            if not setter.intersection(tokens_set_idf):
                continue
            else:   
                count=count+1
        
        return math.log(float((1 + len(self.sentence_tokenizer)) )/ (1 + count))

    def normalised_tfid(self,mytf,myidf,max_tfid):
        for key,values in self.mytf.items():
            normalized=0.4+0.6*(values/max_tfid)
            self.mytf[key]=normalized
            self.tfidf[key]=(self.myidf[key]*self.mytf[key])

    def page_rank(self,G):
        d=0.85
        iterator=100
        tol=1.0e-6
        weight='weight'
        if len(G) == 0:
            return {}

        if not G.is_directed():
            D= G.to_directed()
        else:
            D = G
        W = nx.stochastic_graph(D, weight='weight')
        N = W.number_of_nodes()
        x = dict.fromkeys(W, 1.0 / N)
        p=dict.fromkeys(W, 1.0 / N)

        for _ in range(iterator):
            xlast = x
            x = dict.fromkeys(xlast.keys(), 0)
            for n in x:
                for nbr in W[n]:
                    x[nbr] += d * xlast[n] * W[n][nbr][weight]
                x[n] +=(1.0 - d) * p[n]
            err = sum([abs(x[n] - xlast[n]) for n in x])
            if err < N*tol:
                return x

    def textrank(self,fname):
        f=open(fname,"r")
        document=f.read()
        document = ' '.join(document.strip().split('\n'))
        self.sentence_tokenizer =  nltk.sent_tokenize(document)
        for i in range(len(self.sentence_tokenizer)):
            tokens = self.get_tokens(self.sentence_tokenizer[i])
            tokens_set = set(tokens)
            for word in tokens_set:
                self.mytf[word] = float(tokens.count(word)) / len(tokens)
                self.myidf[word] = self.get_idf(word)
                
            self.normalised_tfid(self.mytf,self.myidf,max(self.mytf.values()))
        out=list()
        word_list=set(self.get_tokens(document))
        for i in range(len(self.sentence_tokenizer)):
            tokens = self.get_tokens(self.sentence_tokenizer[i])
            tokens_set = set(tokens)
            new=list()
            for word in word_list:
                if word in tokens_set:
                    new.append(self.tfidf[word])
                else:
                    new.append(0)
            out.append(new)
        matrix = numpy.matrix(out)
        matrix=matrix*matrix.T
        sA=sparse.csr_matrix(matrix)
        nx_graph = nx.from_scipy_sparse_matrix(sA)
        scores =self.page_rank(nx_graph)
        summary= sorted(((scores[i],s) for i,s in enumerate(self.sentence_tokenizer)),reverse=True)
        f=open("summari.txt","w+")
        for i in range((self.limit*len(summary))/100) :
            sentence=summary[i][1]
            f.write(str(sentence))
        f.close()

class TestApp(App):
    def build(self):
        return Myui()

if __name__ == '__main__':
    TestApp().run()
