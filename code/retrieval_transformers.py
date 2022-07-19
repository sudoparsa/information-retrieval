import json
import time
from preprocess import Preprocess
from sentence_transformers import SentenceTransformer, util
import numpy as np


class TransformerRetrieval():
    def __init__(self,
                 model_path='models/SentenceTransformers/model/',
                 embedding_path='models/SentenceTransformers/doc_embedding.json'):
        self.model = SentenceTransformer(model_path)
        self.doc_embedding = json.load(open(embedding_path))
        self.preprocessor = Preprocess()
    
    def show(self, indexes, show_score=True, scores=None):
        print('\n')
        if show_score:
            print('Similar Papers using Cosine Similarity:')
        else:
            print('Similar Papers:')
        for ix, i in zip(indexes, range(len(indexes))):
            print(f'\n{i}.', end='')
            print(f' {self.doc_embedding["title"][ix]}')
            if show_score:
               print(f'Cosine Similarity : {scores[ix]}')
        print()

    def most_similar(self, query, k):
        query = self.preprocessor.run(query)
        query_embedding = self.model.encode(query)
        embeddings = self.doc_embedding['embedding']
        cosine_scores = util.dot_score(query_embedding, embeddings).detach().cpu().numpy()[0]
        similar_ix = np.argsort(cosine_scores)[::-1][:k]
        return similar_ix, cosine_scores    

    def run(self, query, k=10):
        start_time = time.time()
        print (f'Query: {query}')
        indx, scores = self.most_similar(query, k)
        self.show(indx, scores=scores)
        print()
        print(f'Execution time: {time.time()-start_time}')


transformer_model = TransformerRetrieval()