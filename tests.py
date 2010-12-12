#!/usr/bin/python

''' a set of tests on some stupid-simple data for sanity checking '''

from streamlda import StreamLDA
from util import print_topics
import numpy as n
from matplotlib.pyplot import plot
from pylab import *
import random

# start out with some small docs that have zero vocabulary overlap. also make
# them somewhat intuitive so it's easier to read :)

doc1 = " green grass green grass green grass green grass green grass" 
doc2 = "space exploration space exploration space exploration space exploration" 

num_topics = 2
alpha =1.0/num_topics
eta = 1.0/num_topics
tau0 = 1
kappa =  0.7
slda = StreamLDA(num_topics, alpha, eta, tau0, kappa, sanity_check=False)

num_runs = 200
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

#    if (this_run % 10 == 0):                                                         
#        n.savetxt('lambda-%d.dat' % this_run, slda._lambda.as_matrix())
#        n.savetxt('gamma-%d.dat' % this_run, gamma)

    print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
        (this_run, slda._rhot, perplexity)
    perp.write("%d,%f\n" % (this_run, perplexity))
    perp.flush()
    this_run += 1
perp.close()
print_topics(slda._lambda, 50)


# set up a plot and show the results
xlabel('Run')
ylabel('Perplexity')
title('Perplexity Values - Sanity Check')
plot(range(num_runs), perplexities)
show()


