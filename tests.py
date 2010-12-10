#!/usr/bin/python

''' a set of tests on some stupid-simple data for sanity checking '''

from streamlda import StreamLDA
import numpy as n
from matplotlib.pyplot import plot
from pylab import *
import random

# start out with some small docs that have zero vocabulary overlap. also make
# them somewhat intuitive so it's easier to read :)

def print_topics(lambda_, topn):
    ''' prints the top n most frequent words from each topic in lambda '''
    lambda_mat = lambda_.as_matrix()
    topics = lambda_.num_topics
    for k in xrange(topics):
        # get the probabilities for this topic
        lambdak = list(lambda_mat[k,:]) # a list of counts
        id_to_words = lambda_.indexes
        # lambdak[i] is a probability. id_to_words[i] is the word associated
        # with index i
        wordprobs = [(lambdak[i], id_to_words[i]) for i in xrange(len(lambdak))]
        # sort the probabilities (don't forget, sort() works in place). by
        # default, sort() will sort on the first item in each tuple. 
        wordprobs.sort(reverse=True)
        # print them 
        # feel free to change the "53" here to whatever fits your screen nicely.
        print 'Topic %d' % k
        print '---------------------------'

        for i in xrange(len(wordprobs)): #range(0, 53):
            print '%20s  \t---\t  %.4f' % (wordprobs[i][1], wordprobs[i][0])
        print

doc1 = " green grass green grass green grass green grass green grass" 
doc2 = "space exploration space exploration space exploration space exploration" 

num_topics = 2
alpha =1.0/num_topics
eta = 1.0/num_topics
tau0 = 1024
kappa =  0.7
slda = StreamLDA(num_topics, alpha, eta, tau0, kappa, sanity_check=True)

num_runs = 60
perplexities = []
perp = open('perplexities.dat','w') 
perp.write("Run, Perplexity\n")
this_run = 0
while this_run < num_runs:
    print "Run #%d..." % this_run
    # batch_docs = [random.choice([doc1,doc2]) for i in xrange(batchsize)]
    batch_docs = [doc1, doc2, doc1, doc2, doc2, doc1, doc1, doc1, doc2, doc1]
    (gamma, bound) = slda.update_lambda(batch_docs)
    (wordids, wordcts) = slda.parse_new_docs(batch_docs)
    perwordbound = bound * len(batch_docs) / (slda._D * sum(map(sum, wordcts)))
    perplexity = n.exp(-perwordbound)
    perplexities.append(perplexity)
    print_topics(slda._lambda, 50)

#    if (this_run % 10 == 0):                                                         
#        n.savetxt('lambda-%d.dat' % this_run, slda._lambda.as_matrix())
#        n.savetxt('gamma-%d.dat' % this_run, gamma)

    print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
        (this_run, slda._rhot, perplexity)
    perp.write("%d,%f\n" % (this_run, perplexity))
    perp.flush()
    this_run += 1
perp.close()


# set up a plot and show the results
xlabel('Run')
ylabel('Perplexity')
title('Perplexity Values')
plot(range(num_runs), perplexities)
show()


