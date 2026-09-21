"""Transparent lexical baselines and metrics for the evidence-selection experiment."""
import math
import re
from collections import Counter, defaultdict


def words(text):
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    return re.findall(r'[a-z]+|[0-9]+', text.lower())


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.lengths = []
        self.index = defaultdict(list)
        for i, doc in enumerate(documents):
            tokens = words(doc); self.lengths.append(len(tokens))
            for term, count in Counter(tokens).items(): self.index[term].append((i, count))
        self.n = len(documents); self.average = sum(self.lengths)/max(1,self.n)

    def scores(self, query):
        result = [0.0]*self.n
        for term in set(words(query)):
            posting = self.index.get(term, [])
            idf = math.log(1+(self.n-len(posting)+0.5)/(len(posting)+0.5))
            for i, count in posting:
                norm = self.k1*(1-self.b+self.b*self.lengths[i]/max(1,self.average))
                result[i] += idf*count*(self.k1+1)/(count+norm)
        return result


def ndcg(relevance, order, k=10):
    def dcg(values): return sum((2**r-1)/math.log2(i+2) for i,r in enumerate(values))
    ideal = dcg(sorted(relevance,reverse=True)[:k])
    return dcg([relevance[i] for i in order[:k]])/ideal if ideal else None
