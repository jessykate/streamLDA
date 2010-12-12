#!/usr/bin/python

# onlinewikipedia.py: Demonstrates the use of online VB for LDA to
# analyze a bunch of random Wikipedia articles.
#
# Copyright (C) 2010  Matthew D. Hoffman
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

import cPickle, string, numpy, getopt, sys, random, time, re, pprint

import streamlda
import wikirandom
from util import print_topics

def main():
    """
    Downloads and analyzes a bunch of random Wikipedia articles using
    online VB for LDA.
    """

    # The number of documents to analyze each iteration
    batchsize = 10 #64
    # The number of topics
    K = 10

    if (len(sys.argv) < 2):        
        runs = 50
    else:
        runs = int(sys.argv[1])        

    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    slda = streamlda.StreamLDA(K, 1./K, 1./K, 1., 0.7)
    for iteration in range(0, runs):
        print '-----------------------------------'
        print '         Iteration %d              ' % iteration
        print '-----------------------------------'
        
        # Download some articles
        (docset, articlenames) = \
            wikirandom.get_random_wikipedia_articles(batchsize)
        # Give them to online LDA
        (gamma, bound) = slda.update_lambda(docset)
        # Compute an estimate of held-out perplexity
        wordids = slda.recentbatch['wordids']
        wordcts = slda.recentbatch['wordcts']
        #(wordids, wordcts) = slda.parse_new_docs(docset)
        perwordbound = bound * len(docset) / (slda._D * sum(map(sum, wordcts)))
        print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
            (iteration, slda._rhot, numpy.exp(-perwordbound))

        print_topics(slda._lambda, 10)

if __name__ == '__main__':
    main()
