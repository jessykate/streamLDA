

class Corpus:
  def __init__(self, corpus_name):
    self._name = corpus_name

  def docs(self, num_docs, train=True):
    raise NotImplementedError
