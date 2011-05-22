# onlineldavb.py: Package of functions for fitting Latent Dirichlet
# Allocation (LDA) with online variational Bayes (VB).
#
# Copyright (C) 2011 Jordan Boyd-Graber
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

from glob import glob
from random import sample

from corpora import Corpus

class TwentyNewsCorpus(Corpus):
  def __init__(self, corpus_name, path, deterministic=False):
    self._path = path
    self._deterministic = deterministic

    self._filenames = {}
    print "Searching %s/train/*/*" % self._path
    self._filenames[True] = glob("%s/train/*/*" % self._path)
    self._filenames[False] = glob("%s/test/*/*" % self._path)

    Corpus.__init__(self, corpus_name)

  def docs(self, num_docs, train=True):
    candidates = self._filenames[train]
    if num_docs > 0 and num_docs < len(candidates):
      if self._deterministic:
        selection = candidates[:num_docs]
      else:
        selection = sample(candidates, num_docs)
    else:
      selection = candidates

    return [open(x).read() for x in selection], selection

if __name__ == "__main__":
   c = TwentyNewsCorpus("20news", "data/20_news_date")

   (articles, articlenames) = c.docs(20)
   for ii in range(0, len(articles)):
     print articlenames[ii]
