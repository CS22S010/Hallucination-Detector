# Problem Statement:

Factuality and Hallucination Detection Framework 
Develop a systematic approach to detect and quantify hallucinations in LLM-generated content, with focus on creating verifiable benchmarks and automated detection methods that don't rely solely on ground truth datasets.Technical Background LLMs frequently generate plausible-sounding but factually incorrect information. Current detection methods rely heavily on: 
- Expensive ground truth datasets 
- Post-hoc fact-checking against knowledge bases 
- Manual expert verification 
Challenge: Can we develop reliable hallucination detection using intrinsic model signals, consistency checks, and lightweight verification methods?

## Proposed Solution

We need to detect hallucinations without relying on ground truth dataset.

The following intrinsic signals might indicate hallucination:
- Low token log probability -> This might mean the model is not sure of it's outputs -> high likelihood of hallucination
    - Average token probability, minimum probability and entropy of the probabilities are all features that can indicate this
- Inconsistent or contradictory responses can indicate that the model is hallucinating
    - We use a lightweight NLI model to find contradictions among k generated responses -> compute a score (1 - (number of contradictions/total number of pairs))

The features extracted here can be used along with human judgements to train a ML model (In this example we use LogisticRegression) that can then be used to predict hallucinations.

This approach
  - Does not have have expensive ground truth datasets
  - No post-hoc fact-checking against knowledge bases
  - Manual expert verification will be required to generate a training data set. But this is a one time process, and won't be required all the time

A sample code for this train a model in this approach is given in this repository with toy examples (test.py). An example on how to extract features from model output is also shown using OpenAI API (sample.py).
