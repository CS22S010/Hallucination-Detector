from feature_extractor import HallucinationFeatureExtractor
from hallucination_detector import HallucinationDetector
from joblib import dump

extractor = HallucinationFeatureExtractor()

claim = "Marie Curie won three Nobel Prizes."   # <-- hallucinated claim

token_logprobs = [-1.5, -2.1, -2.3, -1.8, -3.0]  # from your model
token_prob_dists = [
    [0.1, 0.2, 0.7],   # distribution at step 1
    [0.4, 0.4, 0.2],
    [0.05, 0.05, 0.9],
    [0.3, 0.3, 0.4],
    [0.25, 0.25, 0.5],
]

multi_decodes = [
    "Marie Curie won two Nobel Prizes.",
    "Marie Curie won two Nobel Prizes.",
    "Marie Curie won three Nobel Prizes.",
]

retrieved_docs = [
    "Marie Curie is the only person to win Nobel Prizes in two different sciences.",
]

feat = extractor.extract(
    claim_text=claim,
    token_logprobs=token_logprobs,
    token_prob_dists=token_prob_dists,
    multi_decodes=multi_decodes,
)

print("Features:", feat)

claim_B = "The Eiffel Tower is located in Paris."

# Higher logprobs → model is confident
token_logprobs_B = [-0.2, -0.3, -0.1, -0.15, -0.25]

# Distributions concentrated around the chosen token
token_prob_dists_B = [
    [0.80, 0.10, 0.10],
    [0.75, 0.15, 0.10],
    [0.85, 0.10, 0.05],
    [0.70, 0.20, 0.10],
    [0.78, 0.12, 0.10],
]

# Self-consistency — all generations agree
multi_decodes_B = [
    "The Eiffel Tower is located in Paris.",
    "The Eiffel Tower is located in Paris.",
    "The Eiffel Tower is located in Paris."
]

# Retrieval strongly supports it
retrieved_docs_B = [
    "The Eiffel Tower (La tour Eiffel) is a wrought-iron lattice tower in Paris, France."
]

features_B = extractor.extract(
    claim_text=claim_B,
    token_logprobs=token_logprobs_B,
    token_prob_dists=token_prob_dists_B,
    multi_decodes=multi_decodes_B,
)
label_B = 0
print("Features for claim B:", features_B)


train_features = [feat, features_B]  # you would use many
train_labels = [1, 0]       # hallucination label (1=true)

detector = HallucinationDetector(feature_keys=list(feat.keys()))
detector.train(train_features, train_labels)

p = detector.predict_proba(feat)
print("Hallucination probability:", p)

detector.save("hallucination_detector.joblib")

