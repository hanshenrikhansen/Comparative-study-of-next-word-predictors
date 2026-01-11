# ngram_model.py
from collections import defaultdict, Counter
import numpy as np

class NGramModel:
    def __init__(self, n=3):
        """
        n: the 'N' in N-gram
        """
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab_size = None
        self.word_index = None

    def fit(self, tokenized_sentences, word_index):
        self.word_index = word_index
        self.vocab_size = len(word_index) + 1

        for sentence in tokenized_sentences:
            if len(sentence) < 1:
                continue
            padded = [0]*(self.n-1) + sentence
            for i in range(len(sentence)):
                context = tuple(padded[i:i+self.n-1])
                target = padded[i+self.n-1]
                self.ngram_counts[context][target] += 1
                self.context_counts[context] += 1

    def predict_next(self, context, top_k=5):
        if len(context) != self.n-1:
            raise ValueError(f"Context length must be {self.n-1}")
        context = tuple(context)

        counts = self.ngram_counts.get(context, None)
        if counts is None:
            probs = np.ones(self.vocab_size) / self.vocab_size
        else:
            total = self.context_counts[context] + self.vocab_size
            probs = np.ones(self.vocab_size)
            for word_id, count in counts.items():
                probs[word_id] += count
            probs /= total

        top_indices = probs.argsort()[-top_k:][::-1]
        return top_indices, probs[top_indices]

    def get_probability(self, context, word_id):
        context = tuple(context)
        count = self.ngram_counts.get(context, {}).get(word_id, 0)
        total = self.context_counts.get(context, 0) + self.vocab_size
        return (count + 1) / total