#!/usr/bin/python

# dirichlet_words.py: Class to store counts and compute probabilities over
# words in topics. Views process as a three level process. Each topic is drawn
# from a base distribution over words shared among all topics. The word
# distribution backs off to a monkey at a typwriter distribution.
#
# Written by Jordan Boyd-Graber and Jessy Cowan-Sharp
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from nltk import FreqDist
import string, random
import numpy as n

from math import log

CHAR_SMOOTHING = 1 / 10000.

def probability_vector(dims):
    ''' generates a randomized probability vector of the specified dimensions
    (sums up to one) '''
    values = [random.random() for d in xrange(dims)]
    return [v/sum(values) for v in values]

class DirichletWords(object):

  def __init__(self, num_topics, alpha_topic = 1.0, alpha_word = 1.0, 
                max_tables = 50000, sanity_check=False, initialize=False):

    self.max_tables = max_tables
    self._alphabet = FreqDist()
    # store all words seen in a list so they are associated with a unique ID. 
    self.indexes = []
    self._words = FreqDist()

    self.alpha_topic = alpha_topic
    self.alpha_word = alpha_word

    self.num_topics = num_topics
    self._topics = [FreqDist() for x in xrange(num_topics)]

    # the sanity_check flag is for testing only. 
    if initialize and sanity_check == True:
        self.deterministic_seed()
    elif initialize:
        self.initialize_topics()

  def deterministic_seed(self):
    ''' if sanity_check = True, this will seed the topics with enough variance
    to evolve but do so in the most basic and deterministic way possible, so a
    user can follow along each step of the algorithm'''

    chars = "abcdefghijklmnopqrstuvwxyz"
    for i in xrange(3):
      word = random.choice(chars)
      topic_weights = probability_vector(self.num_topics)
      for k in xrange(self.num_topics):
        self.update_count(word, k, topic_weights[k])

  def initialize_topics(self):
    ''' initializes the topics with some random seed words so that they have
        enough relative bias to evolve when new words are passed in.  '''
    # we are going to create some random string from /dev/urandom. to convert
    # them to a string, we need a translation table that is 256 characters. 
    translate_table = (string.letters*5)[:256]
    # /dev/urandom is technically not as random as /dev/random, but it doesn't
    # block. 
    r = open('/dev/urandom')
    # make random 'words' and add them to the topics. they'll never
    # realistically be seen again- which is good since we just want them to
    # seed the bias in the topics. 
    for i in xrange(self.num_topics):
        word_length = random.randint(9,20)
        word = r.read(word_length).translate(translate_table)
        topic_weights = probability_vector(self.num_topics)
        for k in xrange(self.num_topics):
            self.update_count(word, k, topic_weights[k])
    r.close()

  def __len__(self):
    return len(self._words)

  def as_matrix(self):
    ''' Return a matrix of the probabilities of all words over all topics.
        note that because we are using topic_prob(), this is equivalent to he
        expectation of log beta, ie Elogbeta '''
    
    #  XXX TODO we should store this on the fly instead of recomputing it
    #  all the time!

    # create a numpy array here because that's what the e_step in streamLDA
    # expects 
    num_words = len(self.indexes)
    lambda_matrix = n.zeros((self.num_topics, num_words))
    for word_index, word in enumerate(self.indexes):
        topic_weights = [log(self.topic_prob(k, word)) \
                         for k in xrange(self.num_topics)]
        # topic weights for this word-- a column vector. 
        lambda_matrix[:,word_index] = topic_weights
    return lambda_matrix

  def index(self, word):
    if word not in self.indexes:
      self.indexes.append(word)
    return self.indexes.index(word)

  def forget(self, proportion):

    num_tables = len(self._words)      
    number_to_forget = proportion * num_tables
    if num_tables > max_tables:
      number_to_forget += (num_tables - max_tables)
    
    # change this to weight lower probability
    tables_to_forget = random.sample(xrange(num_tables), number_to_forget)
    words = self._words.keys()

    word_id = -1
    for ii in words:
      word_id += 1

      if not word_id in tables_to_forget:
        continue

      count = self._words[ii]
      for jj in self._topics:
        del self._topics[jj][ii]

      for jj in ii:
        self._chars[jj] -= count
      del self._words[ii]

  def seq_prob(self, word):
    val = 1.0

    # Weighted monkeys at typewriter
    for ii in word:
      # Add in a threshold to make sure we don't have zero probability sequences
      val *= max(self._alphabet.freq(ii), CHAR_SMOOTHING) 

    # Normalize
    val /= 2**(len(word))
    return val

  def merge(self, otherlambda, rhot):
    ''' fold the word counts in another DirichletWords object into this
        one, weighted by rhot. assumes self.num_topics is the same for both
        objects. '''
    
    all_words = self._words.keys() + otherlambda._words.keys()
    distinct_words = list(set(all_words))

    # combines the probabilities, with otherlambda weighted by rho, and
    # generates a new count by combining the number of words in the old
    # (current) lambda with the number in the new. here we essentially take
    # the same steps as update_count but do so explicitly so we can weight the
    # terms appropriately. 
    total_words = self._words.N() + otherlambda._words.N()

    self_scale = (1.0-rhot)*total_words/float(self._words.N())
    other_scale = rhot*total_words/float(otherlambda._words.N())

    for word in distinct_words:

      if word not in self.indexes:
        self.indexes.append(word)
        
      # update word counts
      self._words[word] = (self_scale*self._words[word] 
                        + other_scale*otherlambda._words[word])
      
      # update topic counts
      for topic in xrange(self.num_topics):
        self._topics[topic][word] = (self_scale*self._topics[topic][word] 
                                    + other_scale*otherlambda._topics[topic][word])
         
    # update sequence counts
    all_chars = self._alphabet.keys() + otherlambda._alphabet.keys()
    distinct_chars = list(set(all_chars))
 
    for ii in distinct_chars:
      self._alphabet[ii] = (self_scale*self._alphabet[ii] 
                            + other_scale*otherlambda._alphabet[ii])

  def word_prob(self, word):
    return (self._words[word] + self.alpha_word * self.seq_prob(word)) / \
           (self._words.N() + self.alpha_word)

  def topic_prob(self, topic, word):
    return (self._topics[topic][word] + \
            self.alpha_topic * self.word_prob(word)) / \
            (self._topics[topic].N() + self.alpha_topic)

  def update_count(self, word, topic, count):
    # create an index for the word
    if word not in self.indexes:
      self.indexes.append(word)
    # increment the frequency of the word in the specified topic
    self._topics[topic][word] += count
    # also keep a separate frequency count of the number of times this word has
    # appeared, across all documents. 
    self._words[word] += count
    # finally, keep track of the appearance of each character.
    # note that this does not assume any particular character set nor limit
    # recognized characters. if words contain punctuation, etc. then they will
    # be counted here. 
    for ii in word:
      self._alphabet[ii] += count

  def print_probs(self, word):
    print "----------------"
    print word
    for ii in xrange(self.num_topics):
      print ii, self.topic_prob(ii, word)
    print "WORD", self.word_prob(word)
    print "SEQ", self.seq_prob(word)

if __name__ == "__main__":
  test_assignments = [("one",    [0.1, 0.8, 0.1]),
                      ("fish",   [0.0, 0.1, 0.9]),
                      ("two",    [0.1, 0.8, 0.1]),
                      ("fish",   [0.0, 0.2, 0.8]),
                      ("red",    [1.0, 0.0, 0.0]),
                      ("fish",   [0.0, 0.1, 0.9]),
                      ("blue",   [0.25, 0.5, 0.25]),
                      ("fish",   [0.1, 0.5, 0.4])]

  num_topics = len(test_assignments[0][1])
  
  word_prob = DirichletProcessTopics(num_topics)
  for word, phi in test_assignments:
    word_prob.print_probs(word)

    for jj in xrange(num_topics):
      word_prob.update_count(word, jj ,phi[jj])

    word_prob.print_probs(word)
