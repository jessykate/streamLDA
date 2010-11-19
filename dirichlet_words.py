#!/usr/bin/python

# dirichlet_words.py: Class to store counts and compute probabilities over
# words in topics. Views process as a three level process. Each topic is drawn
# from a base distribution over words shared among all topics. The word
# distribution backs off to a monkey at a typwriter distribution.
#
# Written by Jordan Boyd-Graber
# Modifications by Jessy Cowan-Sharp
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

CHAR_SMOOTHING = 1 / 10000.

class DirichletWords(object):

  def __init__(self, num_topics, alpha_topic = 1.0,
               alpha_word = 1.0, max_tables = 50000):

    self._alphabet = FreqDist()
    self._words = FreqDist()

    self.alpha_topic = alpha_topic
    self.alpha_word = alpha_word

    self.num_topics = num_topics
    self._topics = [FreqDist() for x in xrange(num_topics)]

  def __len__(self):
    return len(self._words)

  def forget(self, proportion):

    num_tables = len(self._words)      
    number_to_forget = proportion * num_tables
    if num_tables > max_tables:
      number_to_forget += (num_tables - max_tables)
    
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

  def word_prob(self, word):
    return (self._words[word] + self.alpha_word * self.seq_prob(word)) / \
           (self._words.N() + self.alpha_word)

  def topic_prob(self, topic, word):
    return (self._topics[topic][word] + \
            self.alpha_topic * self.word_prob(word)) / \
            (self._topics[topic].N() + self.alpha_topic)

  def update_count(self, word, topic, count=1):
    # increment the count of the word in the specified topic
    self._topics[topic][word] += count
    # also keep a separate count of the number of times this word has appeared,
    # across all documents. 
    self._words[word] += count
    # finally, keep track of how many times each character has been observed.
    # note that this does not assume any particular character set nor limit
    # recognized characters. if words contain punctuation, etc. then they will
    # be vounted here. 
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
