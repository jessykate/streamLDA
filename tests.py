#!/usr/bin/python

''' a set of tests on some stupid-simple data for sanity checking '''

from streamlda import StreamLDA
import random
import numpy as n

# start out with some small docs that have zero vocabulary overlap. also make
# them somewhat intuitive so it's easier to read :)

doc1 = " green grass bug frog dirt plant tree water road road road tree dirt ant ants walking crawling long seems like ant green leaf cutter leaf cutter"
doc2 = "space exploration dragon spacecraft launch test go shuttle countdown shuttle exploration space space exploration technology technology go for launch delay countdown"

num_topics = 10
alpha =1.0/num_topics
eta = 1.0/num_topics
tau0 = 1024
kappa =  0.7
slda = StreamLDA(num_topics, alpha, eta, tau0, kappa)

num_runs = 100
this_run = 0
batchsize = 10

while this_run < num_runs:
    print "Run #%d..." % num_runs
    batch_docs = [random.choice([doc1,doc2]) for i in xrange(batchsize)]
    (gamma, bound) = slda.update_lambda(batch_docs)
    (wordids, wordcts) = slda.parse_new_docs(batch_docs)
    perwordbound = bound * len(batch_docs) / (slda._D * sum(map(sum, wordcts)))
    print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
        (this_run, slda._rhot, n.exp(-perwordbound))
    lambda_mat = slda._lambda.as_matrix()
    for k in xrange(num_topics):
        # get the probabilities for this topic
        lambdak = list(lambda_mat[k,:]) # a list of counts
        id_to_words = slda._lambda.indexes
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
        for i in range(0, 53):
            print '%20s  \t---\t  %.4f' % (wordprobs[i][1], wordprobs[i][0])
        print
 
    this_run += 1
    raw_input("enter to continue")

