'''
基于歌词来训练模型，用给定起始词，结合长度，来AI歌词生成

1.获取数据，分词，获取词表
2.创建数据集
3.创建网络
4.模型训练
5.模型测试

'''

import torch
import re
import jieba
from torch.utils. data import DataLoader
import torch. nn as nn
import torch.nn. functional as F
import torch.optim as optim
import time

def build_word_sheet():
    #去重
    unique_words, all_words=[],[]
    #便利数据
    for line in open('./data/jaychou_lyrics.txt','r',encoding='utf-8'):
        #分词
        words=jieba.lcut(line)
        # print(f'each line: {words}')
        #
        all_words.append(words)#[[first line],[second line]...]
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    #统计去重后的词汇数量
    word_count=len(unique_words)
    #构建词表，字典形式
    word_to_index={word:i for i,word in enumerate(unique_words)}
    # print(word_to_index)
    #歌词文本用索引表示
    corpus_idx=[]
    for words in all_words:
        #定义变量记录词索引列表(用数字来表示歌词)
        temp=[]
        #获取每一行的词，并获取对应索引
        for word in words:
            temp.append(word_to_index[word])
        #在每行词用空格隔开
        temp.append(word_to_index[' '])
        #获取每个词的索引，添加到corpus——idx中
        corpus_idx.extend(temp)
    #返回结果 唯一词表，词表，去重后的数量，歌词文本用词表索引表示
    return unique_words, word_to_index,word_count,corpus_idx

class Music_Dataset(torch.utils.data.Dataset):
    def __init__(self,num_chars,corpus_idx):
        #文档中词索引
        self.corpus_idx=corpus_idx
        #每句中词的个数
        self.num_chars=num_chars
        #文档数据中词的数量，不去重
        self.word_count=len(self.corpus_idx)
        #句子数量
        self.sen_num=self.word_count//self.num_chars
    #使用len（obj）时，自动调用此方法
    def __len__(self):
        return self.sen_num
    #当使用obj[index]时，自动调用
    def __getitem__(self, idx):
        #idx 词的索引，并修正索引至文档范围
        #确保索引start在合法范围内，避免越界
        start=min(max(idx,0),self.word_count-self.num_chars-1)
        end=start+self.num_chars
        #输入值，从文档中取出 start到end 的索引的词 作为 x
        x=self.corpus_idx[start:end]
        #输出值，网络预测结果
        y=self.corpus_idx[start+1:end+1]
        return torch.tensor(x),torch.tensor(y)


class Model(nn.Module):
    def __init__(self,unique_word_count):#去重的词的数量
        #初始化父类成员
        super().__init__()
        #初始化词嵌入层
        self.ebd=nn.Embedding(unique_word_count,128)
        #循环网络层
        self.rnn=nn.RNN(128,256,1)
        #输出层（全连接层）特征向量维度（与隐藏向量维度一直），词表中的个数
        self.output=nn.Linear(256,unique_word_count)#此表中每个词的概率选最大最为结果

    def forward(self,inputs,hidden):
        #初始化嵌入层
        embd=self.ebd(inputs)
        #rnn处理
        output,hidden=self.rnn(embd.transpose(0,1),hidden)
        output=self.output(output.reshape(shape=(-1,output.shape[-1])))
        return output, hidden

    def init_hidden(self,batch_size):
        #隐藏层初始化：[网络层数，batch，隐藏层的向量维度]
        return torch.zeros(1,batch_size,256)


def train():
    unique_words, word_to_index, unique_word_count, corpus_idx = build_word_sheet()
    dataset=Music_Dataset(16,corpus_idx)

    model=Model(unique_word_count)#预测5703个词的概率
    dataloader=DataLoader(dataset,batch_size=5,shuffle=True)#每批5句，每句32词
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    epochs=10
    for epoch in range(epochs):
        start = time.time()
        #迭代次数就是训练批数
        iter_num,total_loss=0,0.0
        #具体动作
        for x,y in dataloader:
            hidden=model.init_hidden(5)
            output,hidden=model(x,hidden)
            y=torch.transpose(y,0,1).reshape(shape=(-1,))
            loss=criterion(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num+=1
            total_loss+=loss.item()
        print(f'epoch {epoch+1},loss: {total_loss/iter_num:.4f},time: {time.time()-start:2f}s')
    torch.save(model.state_dict(),'./model_music/Textgenerator.pth')


def evaluate(start_word,sen_len):
    #构建词典
    unique_words, word_to_index,word_count,corpus_idx=build_word_sheet()
    #获取模型
    model=Model(word_count)
    #创建模型参数
    model.load_state_dict(torch.load('./model_music/Textgenerator.pth'))
    #获取初始隐藏层
    hidden=model.init_hidden(1)
    #将输入的开始词转换为索引
    word_idx=word_to_index[start_word]
    #列表存放预测的句子
    generator_sentence=[word_idx]#开始由
    for i in range(sen_len):
        #预测模型
        output,hidden=model(torch.tensor([[word_idx]]),hidden)
        #获取预测结果 argmax()从所有结果中找最大值对应的索引
        word_idx=torch.argmax(output)
        #将预测结果添加到列表中
        generator_sentence.append(word_idx)
    #建索引转换成词
    for idx in generator_sentence:
        print(unique_words[idx],end='')





if __name__=='__main__':
    # unique_words,word_to_index,word_count,corpus_idx=build_word_sheet()
    # print(f'number of words:{word_count}')
    # print(f'去重后的词{unique_words}')
    # print(f'index:{word_to_index}')
    # print(f'each word match index:{corpus_idx}')
    #构建数据集
    # dataset=Music_Dataset(5,corpus_idx)
    # print(f'句子9数量{len(dataset)}')
    # model=Model(word_count)
    # for name, parameter in model.named_parameters():
    #     print(name)
    #     print(parameter.shape)
    train()
    evaluate('星星',50)