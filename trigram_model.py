import sys
from collections import defaultdict
import math
import random
import os
import os.path
import copy
"""
COMS W4705 - Natural Language Processing - Fall 2023 
Programming Homework 1 - Trigram Language Models
Leon Gruber, ldg2134
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
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
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
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
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
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        count_uvw = self.trigramcounts[trigram]
        count_uv = self.bigramcounts[trigram[0:2]]

        if trigram[0:2] == ('START','START'):
            count_uv = self.total_sentences

        if count_uvw == 0:
            return 1/len(self.lexicon)

        return count_uvw/count_uv

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        count_uv = self.bigramcounts[bigram]
        count_u = self.unigramcounts[(bigram[0],)]

        # CHECK IF THIS ACUTALLY WORKS
        if bigram[0] == 'START':
            count_u = self.total_sentences

        if count_u == 0:
            return 1/len(self.lexicon)

        return count_uv/count_u
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        count_u = self.unigramcounts[unigram]

        return count_u/self.total_words

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  


    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        #return result
        pass

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        p_w_uv = self.raw_trigram_probability(trigram)
        p_w_v = self.raw_bigram_probability(trigram[1:3])
        p_w = self.raw_unigram_probability((trigram[2],))

        return lambda1 * p_w_uv + lambda2 * p_w_v + lambda3 * p_w
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        probability = 0
        for word in get_ngrams(sentence,3):
            probability += math.log2(self.smoothed_trigram_probability(word))

        return probability

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
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

    model = TrigramModel("hw1_data/brown_train.txt")
    corpus = corpus_reader("hw1_data/brown_train.txt",model.lexicon)
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




    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 


    # Testing perplexity: 
    dev_corpus = corpus_reader("hw1_data/brown_test.txt",model.lexicon)  #sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("Test pp:")
    print(pp)



    # Essay scoring experiment: 
    acc = essay_scoring_experiment('hw1_data/ets_toefl_data/train_high.txt',
                'hw1_data/ets_toefl_data/train_low.txt', "hw1_data/ets_toefl_data/test_high",
                "hw1_data/ets_toefl_data/test_low")
    print("Accuracy: ")
    print(acc)

