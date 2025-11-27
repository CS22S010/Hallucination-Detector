from openai import OpenAI
from feature_extractor import HallucinationFeatureExtractor
from hallucination_detector import HallucinationDetector

client = OpenAI()
prompt = "Explain Marie Curie's Nobel Prizes."

# --- Run the model ---
resp = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role":"user","content":prompt}],
    logprobs=True,
    top_logprobs=5
)

generated = resp.choices[0].message.content
tokens = resp.choices[0].logprobs.content

token_logprobs = [t.logprob for t in tokens]
token_prob_dists = [[e.logprob for e in t.top_logprobs] for t in tokens]
#print(token_prob_dists)

# --- Self-consistency ---
multi_decodes = []
for _ in range(4):
    out = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role":"user","content":prompt}],
        temperature=1.0
    )
    multi_decodes.append(out.choices[0].message.content)

# --- Extract features ---
extractor = HallucinationFeatureExtractor()
features = extractor.extract(
    claim_text=generated,
    token_logprobs=token_logprobs,
    token_prob_dists=token_prob_dists,
    multi_decodes=multi_decodes
)

# --- Predict ---
detector = HallucinationDetector.load("hallucination_detector.joblib")
prob = detector.predict_proba(features)

print("Hallucination probability:", prob)
