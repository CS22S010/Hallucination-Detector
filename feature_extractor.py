import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nliconsistency import NLIConsistency


class HallucinationFeatureExtractor:
    """
    Extracts intrinsic + lightweight features from:
    - model token logprobs
    - multiple decodes (self-consistency)
    - retrieval snippets
    """

    def __init__(self, retrieval_encoder=None):
        # Use TF-IDF for lightweight retrieval similarity (can replace with SBERT)
        self.vectorizer = retrieval_encoder or TfidfVectorizer()


    def avg_logp(self, logprobs):
        return float(np.mean(logprobs))

    def min_logp(self, logprobs):
        return float(np.min(logprobs))

    def entropy(self, prob_dist):
        """
        prob_dist: list of length vocab_size, the token distribution
        """
        p = np.array(prob_dist)
        p = p / (p.sum() + 1e-12)
        return float(-(p * np.log(p + 1e-12)).sum())

    def consistency_score(self, generations):
        """
        generations: list[str]
        Returns fraction of the most common answer.
        """
        nli = NLIConsistency()
        return nli.consistency_score(generations)
        '''
        if len(generations) == 1:
            return 1.0
        unique, counts = np.unique(generations, return_counts=True)
        return float(np.max(counts) / len(generations))
        '''

    def retrieval_similarity(self, claim, retrieved_docs):
        """
        retrieved_docs: List[str] top-k snippets
        Returns max cosine similarity between claim and retrieved docs
        """
        if not retrieved_docs:
            return 0.0

        docs = [claim] + retrieved_docs
        tfidf = self.vectorizer.fit_transform(docs)
        sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        return float(np.max(sims))

    def extract(
        self,
        claim_text: str,
        token_logprobs: list,
        token_prob_dists: list,
        multi_decodes: list,
    ):
        """
        Inputs:
            claim_text: str - the factual claim text
            token_logprobs: list[float] - logp of generated tokens for this claim
            token_prob_dists: list[list[float]] - next-token distributions (optional)
            multi_decodes: list[str] - N different self-consistency generations
            retrieved_docs: list[str] - top-k retrieved snippets

        Returns:
            feature_vector: dict
        """
        features = {}

        # --- intrinsic signals ------------------------------------
        features["avg_logp"] = self.avg_logp(token_logprobs)
        features["min_logp"] = self.min_logp(token_logprobs)

        # entropy of each distribution, averaged
        entropy_list = [self.entropy(dist) for dist in token_prob_dists]
        features["avg_entropy"] = float(np.mean(entropy_list))

        # self-consistency
        features["self_consistency"] = self.consistency_score(multi_decodes)

        # --- lightweight retrieval ---------------------------------
        '''
        features["retrieval_similarity"] = self.retrieval_similarity(
            claim_text, retrieved_docs
        )
        '''
        
        # optional: claim length
        features["claim_length"] = len(claim_text.split())

        return features
