from nltk.corpus import brown
import numpy as np


class HMM():
    def __init__(self):

        return 
        
        
    def supervised_learning(self, all_data):
        self.tag_set = set(tag for sen in all_data for (word, tag) in sen)
        self.word_set = set(word for sen in all_data for (word, tag) in sen)

        self.dict_tag_to_idx = {tag: idx for (idx, tag) in enumerate(self.tag_set)}
        self.dict_word_to_idx = {word: idx for (idx, word) in enumerate(self.word_set)}

        self.dict_idx_to_tag = {idx: tag for (idx, tag) in enumerate(self.tag_set)}
        self.dict_idx_to_word = {idx: word for (idx, word) in enumerate(self.word_set)}

        self.all_data_idx = [[(self.dict_word_to_idx[word], self.dict_tag_to_idx[tag]) for (word, tag) in sen] for sen in all_data]


        ###########################################
        dict_curTag_nextTag_count_forA = {}
        dict_curTag_count_forA = {}

        dict_curTag_curWord_count_forB = {}
        dict_curTag_count_forB = {}

        dict_firstTag_count_forPai = {}


        #######################################
        ##统计出现的次数
        for sen in self.all_data_idx:
        
            ##对pai进行统计
            firstTag = sen[0][1]
            if firstTag not in dict_firstTag_count_forPai.keys():
                dict_firstTag_count_forPai[firstTag] = 0
            dict_firstTag_count_forPai[firstTag] += 1

            for i in range(0, len(sen)):
            
                ##对A进行统计：
                if i < len(sen)-1:
                    cur_tag = sen[i][1]
                    next_tag = sen[i+1][1]
                    if cur_tag not in dict_curTag_count_forA.keys():
                        dict_curTag_count_forA[cur_tag] = 0
                    dict_curTag_count_forA[cur_tag] += 1
                
                    if (cur_tag, next_tag) not in dict_curTag_nextTag_count_forA.keys():
                        dict_curTag_nextTag_count_forA[(cur_tag, next_tag)] = 0
                    dict_curTag_nextTag_count_forA[(cur_tag, next_tag)] += 1

                ##对B进行统计：
                cur_tag = sen[i][1]
                cur_word = sen[i][0]
                if cur_tag not in dict_curTag_count_forB.keys():
                    dict_curTag_count_forB[cur_tag] = 0
                dict_curTag_count_forB[cur_tag] += 1

                if (cur_tag, cur_word) not in dict_curTag_curWord_count_forB.keys():
                    dict_curTag_curWord_count_forB[(cur_tag, cur_word)] = 0
                dict_curTag_curWord_count_forB[(cur_tag, cur_word)] += 1


        #####################################
        ##计算模型参数（A,B,pai）
        self.maxtrix_A = np.zeros((len(self.tag_set), len(self.tag_set)))
        for r in range(0, len(self.tag_set)):
            for c in range(0, len(self.tag_set)):
                if (r, c) in dict_curTag_nextTag_count_forA.keys():
                    self.maxtrix_A[r][c] = dict_curTag_nextTag_count_forA[(r, c)] / dict_curTag_count_forA[r]
                else:
                    self.maxtrix_A[r][c] = 0  

        self.maxtrix_B = np.zeros((len(self.tag_set), len(self.word_set)))
        for r in range(0, len(self.tag_set)):
            for c in range(0, len(self.word_set)):
                if (r, c) in dict_curTag_curWord_count_forB.keys():
                    self.maxtrix_B[r][c] = dict_curTag_curWord_count_forB[(r, c)] / dict_curTag_count_forB[r]
                else:
                    self.maxtrix_B[r][c] = 0  


        self.vec_pai = np.zeros(len(self.tag_set))
        for i in range(0, len(self.tag_set)):
            if i in dict_firstTag_count_forPai.keys():
                self.vec_pai[i] = dict_firstTag_count_forPai[i] / len(self.all_data_idx)

                
                
    def Viterbi(self, sen):
        list_inputWord_idx = [self.dict_word_to_idx[w] for w in sen]

        N = len(self.tag_set)
        M = len(self.word_set)
        T = len(list_inputWord_idx)

        list_tag_P = []
        list_curTag_preTag = []


        ##初始化：
        dict_tag_P = {}
        for h in range(0, N):
            dict_tag_P[h] = self.vec_pai[h] * self.maxtrix_B[h][list_inputWord_idx[0]]
        list_tag_P.append(dict_tag_P)


        ##递推：
        for t in range(1, T):    
        
            dict_tag_P = {}  #一定一定要写的，记住了！！
            dict_curTag_preTag = {}
            for u in range(0, N):

                max_P = 0
                max_h = 0
                for h in range(0, N):
                    tmp = list_tag_P[-1][h] * self.maxtrix_A[h][u]
                    if tmp > max_P:
                        max_P = tmp
                        max_h = h
                dict_tag_P[u] = max_P * self.maxtrix_B[u][list_inputWord_idx[t]]
                dict_curTag_preTag[u] = max_h

            list_tag_P.append(dict_tag_P)
            list_curTag_preTag.append(dict_curTag_preTag)


        ##终止：
        list_outputTag_idx = []
        finalTag_idx = max(list_tag_P[-1].keys(), key=lambda x: list_tag_P[-1][x])
        list_outputTag_idx.append(finalTag_idx)


        ##回溯：
        list_curTag_preTag.reverse()

        for dict_curTag_preTag in list_curTag_preTag:
            preTag_idx = dict_curTag_preTag[finalTag_idx]
            list_outputTag_idx.append(preTag_idx)
            finalTag_idx = preTag_idx

        list_outputTag_idx.reverse()
        list_outputTag = [self.dict_idx_to_tag[idx] for idx in list_outputTag_idx]

        return list_outputTag
        
        
        
if __name__ == '__main__':
    all_data = brown.tagged_sents()
    all_data = [[(word, tag[0]) for (word, tag) in sen] for sen in all_data]
    
    hmm = HMM()
    hmm.supervised_learning(all_data)
    
    sen = ['he', 'is', 'a', 'very', 'great', 'student']
    tags = hmm.Viterbi(sen)
    print(tags)
    
    