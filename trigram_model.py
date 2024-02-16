import sys
from collections import defaultdict
import math
import random
import os
import os.path
import copy
"""
A Trigram Language Model
@Author: Leon Gruber
@Date: December 2023
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  


def get_ngrams(sequence, n):
    sequ = copy.deepcopy(sequence)
    ret_list = []
    i = len(sequ)
    y = 0

    sequ.insert(0,"START")
    if n > 2:
        sequ.insert(0,"START")
        i += 1
    sequ.append("STOP")
    i += 2

    while y + n <= i:
        temp_list = []
        for x in range(n):
            temp_list.append(sequ[y+x])
        y+=1
        ret_list.append(tuple(temp_list))
    return ret_list


class TrigramModel(object):
    
    def __init__(self, corpusfile):
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        self.total_words = 0
        self.total_sentences = 0
    
        # Iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        test = 0

        for sentence in corpus:
            self.total_sentences += 1
            for unigram in get_ngrams(sentence,1):
                if unigram == ('START',):
                    continue
                self.unigramcounts[unigram] += 1
                self.total_words += 1
            for bigram in get_ngrams(sentence,2):
                self.bigramcounts[bigram] += 1
            for trigram in get_ngrams(sentence,3):
                self.trigramcounts[trigram] += 1

        return

    
    def raw_trigram_probability(self,trigram):
        count_uvw = self.trigramcounts[trigram]
        count_uv = self.bigramcounts[trigram[0:2]]

        if trigram[0:2] == ('START','START'):
            count_uv = self.total_sentences

        if count_uvw == 0:
            return 1/len(self.lexicon)

        return count_uvw/count_uv

    
    def raw_bigram_probability(self, bigram):
        count_uv = self.bigramcounts[bigram]
        count_u = self.unigramcounts[(bigram[0],)]

        # CHECK IF THIS ACUTALLY WORKS
        if bigram[0] == 'START':
            count_u = self.total_sentences

        if count_u == 0:
            return 1/len(self.lexicon)

        return count_uv/count_u

    
    def raw_unigram_probability(self, unigram):
        count_u = self.unigramcounts[unigram]

        return count_u/self.total_words


    def smoothed_trigram_probability(self, trigram):
        # Smoothing using linear interpolation
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        p_w_uv = self.raw_trigram_probability(trigram)
        p_w_v = self.raw_bigram_probability(trigram[1:3])
        p_w = self.raw_unigram_probability((trigram[2],))

        return lambda1 * p_w_uv + lambda2 * p_w_v + lambda3 * p_w
        
    def sentence_logprob(self, sentence):
        probability = 0
        for word in get_ngrams(sentence,3):
            probability += math.log2(self.smoothed_trigram_probability(word))

        return probability

    def perplexity(self, corpus):
        sum = 0
        tokens = 0

        for sentence in corpus:
            sum += self.sentence_logprob(sentence)
            tokens += len(sentence) + 1

        sum /= tokens

        return pow(2,-sum)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp_high = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_low = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))

            if pp_high < pp_low:
                correct += 1
            total += 1

        for f in os.listdir(testdir2):
            pp_low = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp_high = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))

            if pp_low < pp_high:
                correct += 1
            total += 1
        
        return correct/total


if __name__ == "__main__":
    #model = TrigramModel(sys.argv[1])

    model = TrigramModel("brown_train.txt")
    corpus = corpus_reader("brown_train.txt",model.lexicon)
    perp = model.perplexity(corpus)
    print("Train pp:")
    print(perp)

    """
    print(model.unigramcounts[('START',)])
    print(model.unigramcounts[('STOP',)])

    print("Bigram prob: ")
    print(model.raw_bigram_probability(("START","the")))
    print(model.bigramcounts[("START","the")])

    print("Trigram prob: ")
    print(model.raw_trigram_probability(("START","START","the")))
    print(model.trigramcounts[("START","START","the")])
    """

    # Testing perplexity: 
    dev_corpus = corpus_reader("brown_test.txt",model.lexicon)  #sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("Test pp:")
    print(pp)

    # Essay scoring experiment: 
    acc = essay_scoring_experiment('train_high.txt','train_low.txt', "test_high","test_low")
    print("Accuracy: ")
    print(acc)

