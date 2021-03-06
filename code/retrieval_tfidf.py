import json
import time
from preprocess import Preprocess


class TFIDF_Retrieval():
    def __init__(self, 
                 title_tfidf_path = 'models/TFIDF/title_tfidf.json',
                 df_info_path = 'models/TFIDF/df_info.json', 
                 abstract_tfidf_path = 'models/TFIDF/abstract_tfidf.json'):
        self.title_tfidf = json.load(open(title_tfidf_path))
        self.abstract_tfidf = json.load(open(abstract_tfidf_path))
        self.df_info = json.load(open(df_info_path))
        self.preprocessor = Preprocess()

    def run_query(self, query, section, k):
        if section == 'title':
            docs = self.title_tfidf
        else:
            docs = self.abstract_tfidf

        query = self.preprocessor.run(query)
        query_terms = query.split()
        N = len(docs)
        results = []
        for i in range(N):
            score = 0
            for term in query_terms:
                tfidf = docs[i].get(term)
                if tfidf == None:
                    continue
                score += tfidf
            results.append((score,i))
        results.sort(key = lambda x: x[0], reverse = True)
        indexs = [x[1] for x in results]
        return indexs[:k]

    def run(self, query, section, k=10):
        start_time = time.time()
        result = self.run_query(query, section, k)
        print(f'Query: {query}')
        self.show(result)
        print(f'Execution time: {time.time()-start_time}')
    
    def show(self, indexes):
        print('\n')
        print('Similar Papers:')
        for ix, i in zip(indexes, range(len(indexes))):
            print(f'\n{i}.', end='')
            print(f' {self.df_info["title"][ix]}')
        print()


tfidf_model = TFIDF_Retrieval()
