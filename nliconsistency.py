import torch
import itertools
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax


class NLIConsistency:
    """
    Lightweight NLI-based consistency calculator using
    microsoft/deberta-v3-small NLI model.
    
    Output score ∈ [0, 1]:
        1.0 → all decodes agree
        0.0 → fully contradictory
    """

    def __init__(self, model_name="cross-encoder/nli-deberta-v3-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # label mapping for NLI models
        self.label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}

    def nli_pair(self, text1, text2):
        """
        Returns NLI label for a pair (text1, text2)
        """
        encoded = self.tokenizer(
            text1, text2,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            logits = self.model(**encoded).logits[0].numpy()

        probs = softmax(logits)
        label_id = probs.argmax()
        return self.label_map[label_id], probs

    def consistency_score(self, texts):
        """
        Compute NLI-based consistency over N decodes.
        
        Metrics:
        - contradiction_rate = (# contradictions) / total_pairs
        - entailment_rate    = (# entailments) / total_pairs
        
        Final Score = entailment_rate * (1 - contradiction_rate)
        """
        if len(texts) <= 1:
            return 1.0

        pairs = list(itertools.combinations(range(len(texts)), 2))
        contradiction_count = 0
        total = len(pairs)

        for i, j in pairs:
            label, _ = self.nli_pair(texts[i], texts[j])
            #print(label)
            if label == "contradiction":
                contradiction_count += 1
        #print("Contradictions, Entailments, Total")
        #print((contradiction_count, entailment_count, total))
        contradiction_rate = float(contradiction_count) / total

        # Hybrid score: reward entailment, penalize contradiction
        score = (1 - contradiction_rate)
        return score
