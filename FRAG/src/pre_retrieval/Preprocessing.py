from utils.Tools import abandon_rels
from pre_retrieval.PreRetrievalModule import PreRetrievalModule
from utils.Query import Query
from utils.SentenceModel import BGEModel, EmbeddingModel
from config import emb_model_dir, rerank_model_dir
# from utils.PPR import personalized_pagerank, rank_ppr_ents


class PreRetrievalModulePPR(PreRetrievalModule):
    def __init__(self, mode="fixed", max_ent=2000, min_ppr=0.005, restart_prob=0.8):
        super().__init__()
        self.mode = mode
        self.max_ent = max_ent
        self.min_ppr = min_ppr
        self.restart_prob = restart_prob

    def process(self, query: Query) -> Query:
        """Use the PPR algorithm to obtain the most relevant subgraphs
        Due to excessive computation, the input data has been preprocessed beforehand and is returned directly
        """
        return query


class PreRetrievalModuleEmb(PreRetrievalModule):
    def __init__(self, window: int = 32, model_dir: str = None):
        super().__init__()
        self.window = window
        model_dir = emb_model_dir if model_dir is None else model_dir
        self.model_dir = model_dir
        self.emb = EmbeddingModel(model_dir)

    def process(self, query: Query) -> Query:
        query = self._process(query)
        return query

    def _process(self, query: Query) -> Query:
        relations = {edge["name"]
                     for edge in query.subgraph.es}
        corpus = list(relations)
        question = query.question
        cosine_scores = self.emb.get_scores(question, corpus)
        sorted_paths = [path for _, path in sorted(
            zip(cosine_scores, corpus), reverse=True)]
        relations = sorted_paths[:self.window]
        entities = set()
        for entity in query.entities:
            try:
                start_vertex = query.subgraph.vs.find(name=entity).index
                entities.add(start_vertex)
            except:
                continue
        for edge in query.subgraph.es:
            if edge["name"] in relations:
                entities.add(edge.source)
                entities.add(edge.target)
        query.subgraph = query.subgraph.subgraph(entities)
        return query


class PreRetrievalModuleBGE(PreRetrievalModule):
    def __init__(self, window: int = 32, model_dir: str = None):
        super().__init__()

        self.window = window
        model_dir = rerank_model_dir if model_dir is None else model_dir
        self.model_dir = model_dir
        self.bge = BGEModel(model_dir)

    def process(self, query: Query) -> Query:
        query = self._process(query)
        return query

    def _process(self, query: Query) -> Query:
        relations = {edge["name"]
                     for edge in query.subgraph.es if not abandon_rels(edge["name"])}
        corpus = list(relations)
        question = query.question
        cosine_scores = self.bge.get_scores(question, corpus)
        sorted_paths = [path for _, path in sorted(
            zip(cosine_scores, corpus), reverse=True)]
        relations = sorted_paths[:self.window]
        # export subgraph with new relations
        entities = set()
        for entity in query.entities:
            try:
                start_vertex = query.subgraph.vs.find(name=entity).index
                entities.add(start_vertex)
            except:
                continue
        for edge in query.subgraph.es:
            if edge["name"] in relations:
                entities.add(edge.source)
                entities.add(edge.target)
        query.subgraph = query.subgraph.subgraph(entities)
        return query
