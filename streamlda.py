# onlineldavb.py: Package of functions for fitting Latent Dirichlet
# Allocation (LDA) with online variational Bayes (VB).
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

import sys, re, time, string
import numpy as n
from scipy.special import gammaln, psi
from dirichlet_words import DirichletWords
import time
from nltk.corpus import stopwords

n.random.seed(100000001)
meanchangethresh = 0.001

class ParameterError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def dirichlet_expectation(alpha):
    """
    alpha is a W by K dimensional matric. 
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    Returns a W x K matrix. 
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

class StreamLDA:
    """
    Implements stream-based LDA as an extension to online Variational Bayes for
    LDA, as described in (Hoffman et al. 2010).  """

    def __init__(self, K, alpha, eta, tau0, kappa, sanity_check=False):
        """
        Arguments:
        K: Number of topics
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """

        if not isinstance(K, int):
            raise ParameterError

        # set the model-level parameters
        self._K = K
        self._alpha = alpha
        self._eta = eta
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self.sanity_check = sanity_check
        # number of documents seen *so far*. Updated each time a new batch is
        # submitted. 
        self._D = 0

        # number of batches processed so far. 
        self._batches_to_date = 0

        # cache the wordids and wordcts for the most recent batch so they don't
        # have to be recalculated when computing perplexity
        self.recentbatch = {'wordids': None, 'wordcts': None}

        # Initialize lambda as a DirichletWords object which has a non-zero
        # probability for any character sequence, even those unseen. 
        self._lambda = DirichletWords(self._K, sanity_check=self.sanity_check, initialize=True)
        self._lambda_mat = self._lambda.as_matrix()

        # set the variational distribution q(beta|lambda). 
        self._Elogbeta = self._lambda_mat # num_topics x num_words
        self._expElogbeta = n.exp(self._Elogbeta) # num_topics x num_words
        
    def parse_new_docs(self, new_docs):
        """
        Parse a document into a list of word ids and a list of counts,
        or parse a set of documents into two lists of lists of word ids
        and counts.

        Arguments: 
        new_docs:  List of D documents. Each document must be represented as
                   a single string. (Word order is unimportant.) 

        Returns a pair of lists of lists:

        The first, wordids, says what vocabulary tokens are present in
        each document. wordids[i][j] gives the jth unique token present in
        document i. (Don't count on these tokens being in any particular
        order.)

        The second, wordcts, says how many times each vocabulary token is
        present. wordcts[i][j] is the number of times that the token given
        by wordids[i][j] appears in document i.
        """

        # if a single doc was passed in, convert it to a list. 
        if type(new_docs) == str:
            new_docs = [new_docs,]
            
        D = len(new_docs)
        print 'parsing %d documents...' % D

        wordids = list()
        wordcts = list()
        for d in range(0, D):
            # remove non-alpha characters, normalize case and tokenize on
            # spaces
            new_docs[d] = new_docs[d].lower()
            new_docs[d] = re.sub(r'-', ' ', new_docs[d])
            new_docs[d] = re.sub(r'[^a-z ]', '', new_docs[d])
            new_docs[d] = re.sub(r' +', ' ', new_docs[d])
            words = string.split(new_docs[d])
            doc_counts = {}
            for word in words:
                # skip stopwords 
                if word in stopwords.words('english'):
                    continue
                # index returns the unique index for word. if word has not been
                # seen before, a new index is created. We need to do this check
                # on the existing lambda object so that word indices get
                # preserved across runs. 
                wordindex = self._lambda.index(word)
                doc_counts[wordindex] = doc_counts.get(wordindex, 0) + 1

            # if the document was empty, skip it. 
            if len(doc_counts) == 0:
                continue

            # wordids contains the ids of words seen in this batch, broken down
            # as one list of words per document in the batch. 
            wordids.append(doc_counts.keys())
            # wordcts contains counts of those same words, again per document. 
            wordcts.append(doc_counts.values())
            # Increment the count of total docs seen over all batches. 
            self._D += 1
       
        # cache these values so they don't need to be recomputed. 
        self.recentbatch['wordids'] = wordids
        self.recentbatch['wordcts'] = wordcts

        return((wordids, wordcts))

    def do_e_step(self, docs):
        """
        Given a mini-batch of documents, estimates the parameters
        gamma controlling the variational distribution over the topic
        weights for each document in the mini-batch.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns a tuple containing the estimated values of gamma,
        as well as sufficient statistics needed to update lambda.
        """
        # This is to handle the case where someone just passes in a single
        # document, not in a list.
        if type(docs) == str: docs = [docs,]
       
        (wordids, wordcts) = self.parse_new_docs(docs)
        # don't use len(docs) here because if we encounter any empty documents,
        # they'll be skipped in the parse step above, and then batchD will be
        # longer than wordids list. 
        batchD = len(wordids)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1*n.random.gamma(100., 1./100., (batchD, self._K)) # batchD x K
        Elogtheta = dirichlet_expectation(gamma) # D x K
        expElogtheta = n.exp(Elogtheta)

        # create a new_lambda to store the stats for this batch
        new_lambda = DirichletWords(self._K, sanity_check=self.sanity_check)

        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        for d in range(0, batchD):
            print 'Updating gamma and phi for document %d in batch' % d
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :] # K x 1
            expElogthetad = expElogtheta[d, :] # k x 1 for this D. 
            # make sure exp/Elogbeta is initialized for all the needed indices. 
            self.Elogbeta_sizecheck(ids)
            expElogbetad = self._expElogbeta[:, ids] # dims(expElogbetad) = k x len(doc_vocab)
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100

            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad
                # In these steps, phi is represented implicitly to save memory
                # and time.  Substituting the value of the optimal phi back
                # into the update for gamma gives this update. Cf. Lee&Seung
                # 2001.
                gammad = self._alpha + expElogthetad * \
                    n.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step. Updates the statistics only for words
            # in ids list, with their respective counts in cts (also a list).
            # the multiplying factor from self._expElogbeta
            # lambda_stats is basically phi multiplied by the word counts, ie
            # lambda_stats_wk = n_dw * phi_dwk
            # the sum over documents shown in equation (5) happens as each
            # document is iterated over. 

            # lambda stats is K x len(ids), while the actual word ids can be
            # any integer, so we need a way to map word ids to their
            # lambda_stats (ie we can't just index into the lambda_stats array
            # using the wordid because it will be out of range). so we create
            # lambda_data, which contains a list of 2-tuples of length len(ids). 
            # the first tuple item contains the wordid, and the second contains
            # a numpy array with the statistics for each topic, for that word.

            if update_topics:
              lambda_stats = n.outer(expElogthetad.T, cts/phinorm) * expElogbetad
              lambda_data = zip(ids, lambda_stats.T)
              for wordid, stats in lambda_data:
                  word = self._lambda.indexes[wordid]
                  for topic in xrange(self._K):
                      stats_wk = stats[topic]
                      new_lambda.update_count(word, topic, stats_wk)

        return((gamma, new_lambda))

    def update_lambda(self, docs):
        """
        The primary function called by the user. First does an E step on the
        mini-batch given in wordids and wordcts, then uses the result of that E
        step to update the variational parameter matrix lambda.

        docs is a list of D documents each represented as a string. (Word order
        is unimportant.) 

        Returns gamma, the parameters to the variational distribution over the
        topic weights theta for the documents analyzed in this update.

        Also returns an estimate of the variational bound for the entire corpus
        for the OLD setting of lambda based on the documents passed in. This
        can be used as a (possibly very noisy) estimate of held-out likelihood.  
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._batches_to_date, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma, new_lambda) = self.do_e_step(docs)
        # Estimate held-out likelihood for current values of lambda.
        bound = self.approx_bound(gamma)
        # Update lambda based on documents.
        self._lambda.merge(new_lambda, rhot)
        # update the value of lambda_mat so that it also reflect the changes we
        # just made. 
        self._lambda_mat = self._lambda.as_matrix()
        
        # do some housekeeping - is lambda getting too big?
        oversize_by = len(self._lambda._words) - self._lambda.max_tables
        if oversize_by > 0:
            percent_to_forget = oversize_by/len(self._lambda._words)
            self._lambda.forget(percent_to_forget)

        # update expected values of log beta from our lambda object
        self._Elogbeta = self._lambda_mat
#        print 'self lambda mat'
#        print self._lambda_mat
#        print 'self._Elogbeta from lambda_mat after merging'
#        print self._Elogbeta
        self._expElogbeta = n.exp(self._Elogbeta)
#        print 'and self._expElogbeta'
        self._expElogbeta
#        raw_input()
        self._batches_to_date += 1

        return(gamma, bound)

    def Elogbeta_sizecheck(self, ids):
        ''' Elogbeta is initialized with small random values. In an offline LDA
        setting, if a word has never been seen, even after n iterations, its value in
        Elogbeta would remain at this small random value. However, in offline LDA,
        the size of expElogbeta in the words dimension is always <= the number
        of distinct words in some new document. In stream LDA, this is not
        necessarily the case. So we still make sure to use the previous
        iteration's values of Elogbeta, but where a new word appears, we need
        to seed it. That is done here.  '''
        
        # since ids are added sequentially, then the appearance of some id = x
        # in the ids list guarantees that every ID from 0...x-1 also exists.
        # thus, we can take the max value of ids and extend Elogbeta to that
        # size. 
        columns_needed = max(ids)+ 1
        current_columns = self._Elogbeta.shape[1]
        if columns_needed > current_columns:
            self._Elogbeta = n.resize(self._Elogbeta, (self._K, columns_needed))
            # fill the new columns with appropriately small random numbers
            newdata = n.random.random((self._K, columns_needed-current_columns))
            newcols = range(current_columns, columns_needed)
            self._Elogbeta[:,newcols] = newdata
            self._expElogbeta = n.exp(self._Elogbeta)

    def approx_bound(self, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """
        wordids = self.recentbatch['wordids']
        wordcts = self.recentbatch['wordcts']
        batchD = len(wordids)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in range(0, batchD):
            gammad = gamma[d, :]
            ids = wordids[d]
            cts = n.array(wordcts[d])
            phinorm = n.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(cts * phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma)*Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha*self._K) - gammaln(n.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / batchD

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta-self._lambda.as_matrix())*self._Elogbeta)
        score = score + n.sum(gammaln(self._lambda_mat) - gammaln(self._eta))
        score = score + n.sum(gammaln(self._eta*len(self._lambda)) - 
                              gammaln(n.sum(self._lambda_mat, 1)))

        return(score)

