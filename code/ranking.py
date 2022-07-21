import json

pagerank_result_path='models/Ranking/pagerank_result.json'
hits_result_path='models/Ranking/hits.json'

pagerank_result = json.load(open(pagerank_result_path))
hits_result = json.load(open(hits_result_path))