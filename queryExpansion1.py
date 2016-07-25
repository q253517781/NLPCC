#coding=utf8
import pandas as pd 
import os
import subprocess
import collections
import numpy as np 
from nltk.metrics import edit_distance
from gensim import models
import jieba
import pynlpir
import urllib2,math
import random,time,chardet
import cPickle as pickle
from gensim.models.doc2vec import TaggedLineDocument,Doc2Vec
# model= Doc2Vec.load("Doc2vec.model")
# model_vocab=model.vocab.keySet()
from bs4 import BeautifulSoup

#words=open("Chinese_Stopwords.txt").read().split()
words=open("chStopWordsSimple.txt").read().split()
stopwords={word.decode("utf-8") for word in words}
#print "#".join(stopwords)
pynlpir.open()
#pynlpir.nlpir.ImportUserDict("user.dict")
release=False
qa_path="nlpcc-iccpol-2016.dbqa.training-data"
pkl_file="knowledge.pkl"
entityMap= pickle.load(open(pkl_file,"rb"))


def evaluation(modelfile,resultfile="result.text",input=qa_path):
	cmd="test.exe " + " ".join([input,modelfile,resultfile])
	print modelfile[19:-6]+":" 
	subprocess.call(cmd, shell=True)


def cut_sentence(row):
	#print "#".join(cut(row,done=False))
	return "#".join(cut(row,done=False))


def done(filename="done.csv"):
	if os.path.exists(filename):
		return pd.read_csv(filename,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	df=pd.read_csv(qa_path,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	df["question"]=df["question"].apply(cut_sentence)
	df["answer"]=df["answer"].apply(cut_sentence)
	print df

	df.to_csv(filename,index=False,header =False,sep="\t",encoding="utf-8")
	return df


def cut(sentence,done=True):
	if done:
		return [word for word in str(sentence).split("#")]
	try:
		words= pynlpir.segment(sentence, pos_tagging=False)
	#print question
	except:
		words= jieba.cut(sentence)
	#print "$".join(words)
	words=[word for word in words if word not in stopwords]
	return words


def loadData(filename="done.csv"):
	if os.path.exists(filename):
		return pd.read_csv(filename,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	df=pd.read_csv(qa_path,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	df["question"]=df["question"].apply(cut_sentence)
	df["answer"]=df["answer"].apply(cut_sentence)
	print df
	df.to_csv(filename,index=False,header =False,sep="\t",encoding="utf-8")
	return df


def getHtmlbyQuestion(question):
	try:
		filename="train_baike/"+entityMap[question]+".txt"
	except:
		return None
	if os.path.exists(filename):
		html=open(filename).read()
		return html
	else :
		return None


def getProperties(html):
	soup=BeautifulSoup(html, "lxml")
	key=soup.select('.basicInfo-block dt')
	value=soup.select('.basicInfo-block dd')
	properties={}
	if len(key)==len(value):
		for i in range(len(key)):
			name="".join(key[i].text.split())
			p_Value="".join(value[i].text.split())
			#print "%s:%s"%(name,p_Value)
			properties[name]=" ".join(pynlpir.segment(p_Value, pos_tagging=False))
	return properties


def getDiscription(html):
	soup=BeautifulSoup(html,"lxml")
	tag=soup.select('.para')
	dislist=""
	for i in range(len(tag)):
		discription="".join(tag[i].text.split())
		dislist+=(discription)#描述文本变为一个长字符串
	wordss=open("chStopWordsSimple.txt").read().split()#去掉停用词
	stopwords={word.decode("utf-8") for word in wordss}
	try:
		words= pynlpir.segment(dislist, pos_tagging=False)
		#print question
	except:
		words= jieba.cut(dislist)
		#print "$".join(words)
	words=[word for word in words if word not in stopwords]#把描述分词去掉停用词后的列表
	return words


def matchKeyWords(questKeyWord,words):
	#questKeyWord=["喻虹渊","周渝民","台湾"]#查询关键词
	windowWords={}
	for x in range(len(questKeyWord)):#x为查询关键词的索引
		for t in range(len(words)):#t为描述discription中所有分词的索引
			if words[t]==questKeyWord[x]:#查询中的关键字与描述中的分词相匹配，找出描述中相应关键词的位置
				#print "%s"%(words[t])
				if t<5:#关键字在前5个位置上时，取其前面几个和后5个词
					for z in range(0,t+5):
						windowWords.setdefault(words[z],0)
						windowWords[words[z]]= windowWords[words[z]] +   (5-math.fabs(z-t))*1.0/5 
						#print "%s:%s"%(z,words[z])
				if 5<t&t<(len(words)-5):#关键字不在前5个位置上时，取前后各5个词
					for z in range(t-4,t+5):
						windowWords.setdefault(words[z],0)
						windowWords[words[z]]= windowWords[words[z]] +   (5-math.fabs(z-t))*1.0/5 
				if t>(len(words)-5):
					for z in range(t-4,len(words)):
						windowWords.setdefault(words[z],0)
						windowWords[words[z]]= windowWords[words[z]] +   (5-math.fabs(z-t))*1.0/5 
	#print len(windowWords)
	for k,v in windowWords.items():
		print "%s -> %f" % (k,v)
	return windowWords#匹配出的{discriptions'扩展关键词：权重}字典

def proExpansion(questKeyWord,properties):#从属性里面匹配关键词并赋权重
    prolist=list(properties)
    prolist2=list(properties.values())#把{属性：关键词}字典变成一个关键词列表prolist
    prolist.extend(prolist2)
    #print ",".join(prolist)
    #print len(prolist)
    proWindowWords={}
    for x in range(len(questKeyWord)):#属性关键词与查询关键词相匹配 
        for t in range(len(prolist)):
            if prolist[t]==questKeyWord[x]:
                if len(prolist)<3:#给关键词赋权重  
                        proWindowWords.setdefault(prolist[z],1)
                if 3<t&t<(len(prolist)-3):
                    for z in range(t-2,t+3):
                        proWindowWords.setdefault(prolist[z],0)
                        proWindowWords[prolist[z]]=proWindowWords[prolist[z]]+(3-math.fabs(z-t))*2.0/3
                if t>3&t>(len(prolist)-3):  
                    for z in range(t-2,len(prolist)):
                        proWindowWords.setdefault(prolist[z],0)
                        proWindowWords[prolist[z]]=proWindowWords[prolist[z]]+(3-math.fabs(z-t))*2.0/3
    for k,v in proWindowWords.items():
        print "%s:%f"%(k,v) 
    return proWindowWords#匹配出的{properties'扩展关键词：权重}字典


def calScore(q,a,dictMerged):
	total=0.000
	question=cut(q)
	answer=cut(a)
	#print dictMerged.keys()
	exit()
	#question.append(getEntityByName(question))#添加问题中的实体到问题关键词列表question
	questionDict={}
	answerDict={}
	for x in question:
		questionDict.setdefault(x,1)
	for x in answer:
		answerDict.setdefault(x,1)#把Q和A都变为字典，权重都赋1 
	for k1 in questionDict:
		for k2 in dictMerged:
			if k1==k2:#将Q和qe中相匹配的词权重相加
				dictMerged[k2]=questionDict[k1]+dictMerged[k2]
	for k3 in answerDict:
		for k2 in dictMerged:
			if k3==k2:
				dictMerged[k2]=answerDict[k3]+dictMerged[k2]#将Q+qe中与answer中匹配的词权重相加
				total=total+dictMerged[k]
	return total



def qe(row):
	question=row["question"]
	html=getHtmlbyQuestion(question)
	if html==None:
		return 0 # test
	properties=getProperties(html)
	discription=getDiscription(html)

	keywords=pynlpir.get_key_words(question, weighted=False)  #True
	weightedDict1=matchKeyWords(keywords,discription)
	weightedDict2=proExpansion(keywords,properties)
	dictMerged=weightedDict1.copy()
	dictMerged.update(weightedDict2)#所有{扩展关键词：权重}字典

	answer=row["answer"]
	return calScore(question,answer,dictMerged)


def word_overlap(row):

	question=cut(row["question"]) 
	answer=cut(row["answer"])
	overlap= set(answer).intersection(set(question)-stopwords) 
	return len(overlap)


def main():
	df=done()
	methods={"overlap":word_overlap,"qe":qe}
	for name,method in methods.items():
		df[name]=df.apply(method,axis=1)
		model_file="model/train.QApair."+name+".score"
		df[name].to_csv(model_file,index=False,sep="\t")
		evaluation(model_file,input=qa_path)


if __name__=="__main__":
	main()