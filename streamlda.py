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

# XXX QUESTIONS
# 1. should wordids and wordcts use the cts directly or the probabilities from
# _lambda?

import sys, re, time, string
import numpy as n
from scipy.special import gammaln, psi
from dirichlet_words import DirichletWords

n.random.seed(100000001)
meanchangethresh = 0.001

def dirichlet_expectation(alpha):
    """
    alpha is a W by K dimensional matric. 
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    Returns a W x K matrix. 
    """

    # len(alpha) for an n.random.gamma obj is k, or num topics. 
    if (len(alpha) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

class StreamLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, K, alpha, eta, tau0, kappa):
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

        # set the model-level parameters
        self._K = K
        self._alpha = alpha
        self._eta = eta
        self._tau0 = tau0 + 1
        self._kappa = kappa

        # number of documents seen *so far*. Updated each time a new batch is
        # submitted. 
        self._D = 0

        # number of batches processed so far. 
        self._batches_to_date = 0

        # Initialize lambda as a DirichletWords object which has a non-zero
        # probability for any character sequence, even those unseen. 
        self._lambda = DirichletWords(self._K)

        # Current estimate of the k-dimensional topic proportions theta, which
        # in general are estimated from gamma, and summed over all
        # documents and normalized to get a propability over topics. Note that
        # here we initialize theta to a symmetric prior.
        self._theta = n.ones(self._K)*1./(self._K)

        # number of words in the vocabulary *so far*. initially some small
        # non-zero number based on the random topic initialization. 
        self._W = len(self._lambda)

        # do NOT initialize these here-- do that after we parse the words for
        # this batch so that the dimensions match. 
        self._Elogbeta = None #dirichlet_expectation(self._lambda)
        self._expElogbeta = None #n.exp(self._Elogbeta)

        
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
        # increment the count of total docs seen over all batches. 
        self._D += D

        wordids = list()
        wordcts = list()
        for d in range(0, D):
            print 'parsing document %d...' % d
            new_docs[d] = new_docs[d].lower()
            new_docs[d] = re.sub(r'-', ' ', new_docs[d])
            new_docs[d] = re.sub(r'[^a-z ]', '', new_docs[d])
            new_docs[d] = re.sub(r' +', ' ', new_docs[d])
            words = string.split(new_docs[d])
            batch_counts = {}
            for word in words:
                # use the current topic mixing weights for this new observation 
                for k in self._K:
                    self._lambda.update_count(word, k, self._theta[k])
                batch_counts[w] = batch_counts.get(w, 0) + self._vocab._words[word]

            # wordids contains the ids of words seen in this batch, broken down
            # as one list of words per document in the batch. 
            wordids.append(batch_counts.keys())
            # wordcts contains counts of those same words, again per document. 
            wordcts.append(batch_counts.values())
        
        # XXX TODO check this is in the right place:
        # set the variational distribution q(beta|lambda). in onlineLDA, this
        # is done in init(), but we need Elogbeta to be the same dimension as
        # lambda, so instead we initialize it here from lambda. note that
        # self._Elogbeta is overwritten with a new value (as a function of
        # lambda) each batch. 
        # XXX alternatively, we could initialize this to random values like was
        # done in onlineLDA. 
        self._Elogbeta = dirichlet_expectation(self._lambda.as_matrix())
        self._expElogbeta = n.exp(self._Elogbeta)

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
        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if type(docs) == str: docs = [docs,]
       
        (wordids, wordcts) = self.parse_new_docs(docs)
        batchD = len(docs)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1*n.random.gamma(100., 1./100., (batchD, self._K)) # batchD x K
        Elogtheta = dirichlet_expectation(gamma) # D x K
        expElogtheta = n.exp(Elogtheta)
        
        sstats = n.zeros(self._lambda.shape)

        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        for d in range(0, batchD):
            print 'Batch document %d' % d
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :] # K x 1
            expElogthetad = expElogtheta[d, :] # k x 1 for this D. 
            expElogbetad = self._expElogbeta[:, ids] # k's for this d for each word
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100

            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
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
            sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        # save the normalized sum over all documents as the current estimate of
        # the topic proportions, theta:
        self._theta = n.sum(gamma, 0)

        return((gamma, sstats))

    def update_lambda(self, docs):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.

        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._batches_to_date, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma, sstats) = self.do_e_step(docs)
        # Estimate held-out likelihood for current values of lambda.
        bound = self.approx_bound(docs, gamma)
        # Update lambda based on documents.
        self._lambda = self._lambda * (1-rhot) + \
            rhot * (self._eta + self._D * sstats / len(docs))
        self._Elogbeta = dirichlet_expectation(self._lambda.as_matrix())
        self._expElogbeta = n.exp(self._Elogbeta)
        self._batches_to_date += 1

        return(gamma, bound)

    def approx_bound(self, docs, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """

        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if (type(docs).__name__ == 'string'):
            temp = list()
            temp.append(docs)
            docs = temp

        (wordids, wordcts) = self.parse_new_docs(docs)
        batchD = len(docs)

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
#             oldphinorm = phinorm
#             phinorm = n.dot(expElogtheta[d, :], self._expElogbeta[:, ids])
#             print oldphinorm
#             print n.log(phinorm)
#             score += n.sum(cts * n.log(phinorm))

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma)*Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha*self._K) - gammaln(n.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / len(docs)

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta-self._lambda)*self._Elogbeta)
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + n.sum(gammaln(self._eta*self._W) - 
                              gammaln(n.sum(self._lambda, 1)))

        return(score)
        
# XXX TODO add a method to the class for classification of new documents. documents
