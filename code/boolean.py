import json
import time
from preprocess import Preprocess

class BooleanRetrieval():
    def __init__(self, 
                 inverted_indices_path='models/Boolean/inverted_indices.json',
                 df_info_path='models/Boolean/df_info.json'):
        self.inverted_indices = json.load(open('models/Boolean/inverted_indices.json'))
        self.df_info = json.load(open(df_info_path))
        self.preprocessor = Preprocess()
    
    def bool_query(self, query, section, k):
        if section == 'title':
            query = self.preprocessor.run(query)
        elif section == 'author':
            query = self.preprocessor.simple(query)
        doc_list = []
        for word in query.split():
            if (word in self.inverted_indices[section].keys()):
                doc_list += self.inverted_indices[section][word]
        doc_list = sorted(doc_list, key=doc_list.count, reverse=True)
        doc_list = list(dict.fromkeys(doc_list))
        return doc_list[:k]

    def run(self, query, section, k=10):
        start_time = time.time()
        result = self.bool_query(query, section, k)
        print(f'Query: {query}')
        self.show(result, show_score=False)
        print(f'Execution time: {time.time()-start_time}')
    
    def show(self, indexes, show_score=True, scores=None):
        print('\n')
        if show_score:
            print('Similar Papers using Cosine Similarity:')
        else:
            print('Similar Papers:')
        for ix, i in zip(indexes, range(len(indexes))):
            print(f'\n{i}.', end='')
            print(f' {self.df_info["title"][ix]}')
            if show_score:
               print(f'Cosine Similarity : {scores[ix]}')
        print()


boolean_model = BooleanRetrieval()
