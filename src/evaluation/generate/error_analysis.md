# Phase 4 — Generation Evaluation & Error Analysis

## 1. Evaluation Summary

| Metric | Value |
|---|---|
| Total Queries | 26 |
| Avg Semantic Similarity | 0.5592 |
| Avg Faithfulness | 0.3905 |
| Good Answers (both ≥ 0.5) | 6 |
| Poor Answers | 20 |
| Non-English Responses | 0 |

---

## 2. Overview

This report evaluates the quality of generated answers in the Mini-RAG system. Each query is evaluated on two metrics:

- **Semantic Similarity** — how closely the generated answer matches the ground truth (0-1)
- **Faithfulness** — how much of the generated answer is grounded in the retrieved context (0-1)

A low faithfulness score indicates hallucination — the LLM generated content not present in the retrieved chunks.

---

## 3. Edge Case Analysis

### Edge Case 1 — LLM Hallucination Despite Context

**Pattern:** Low faithfulness score with high-sounding answer

The LLM (qwen2.5:3b) frequently generated detailed answers that sounded correct but were not grounded in the retrieved chunks. This occurs because small models tend to ignore the 'use only the context' instruction and fall back to their training knowledge. The faithfulness score captures this — a score near 0 means the answer shares almost no vocabulary with the retrieved context.

**Example:**
- Query: *'What does a DevOps Engineer do?'*
- Retrieved context: CI/CD pipeline bullet points from one chunk
- Generated: A detailed 8-point answer about DevOps not present in the chunk
- Faithfulness: 0.086 — weak grounding

**Proposed Fix:** Use a larger model with stronger instruction-following capability, or add explicit output constraints like 'Answer in maximum 2 sentences using only the context.'

### Edge Case 2 — Semantic Mismatch with Ground Truth

Even when the LLM produces a factually correct answer, the semantic similarity score can be lower than expected because the generated answer uses different phrasing than the ground truth. This was addressed by switching from character-level SequenceMatcher to embedding-based cosine similarity using nomic-embed-text, which compares meaning rather than exact wording.

**Fix Applied:** Replaced SequenceMatcher with embedding cosine similarity — avg semantic similarity jumped from 0.0487 to 0.548.

### Edge Case 3 — Empty Context Leading to Default Response

When the Qdrant collection is empty or the working directory is wrong, NlpController returns 'No relevant documents found in the database.' for every query. This causes semantic similarity scores around 0.2-0.4 and faithfulness of 0.0. This was observed during initial runs before the working directory fix was applied.

**Fix Applied:** Added `os.chdir(SRC_DIR)` at the top of the evaluation script to ensure the correct working directory is set before Qdrant initializes.

### Edge Case 4 — Non-English Output (Language Switch Bug)

**Root Cause:** `qwen2.5:3b` is developed by Alibaba Cloud and defaults to Chinese when it cannot find relevant context in the retrieved chunks, even when explicitly instructed to respond in English only. This is a known limitation of this specific model.

**Proposed Fix:** Switch to a model without Chinese language bias such as `mistral` or `llama3`. These models reliably follow the English-only instruction regardless of context quality.

---

## 4. Poor Generation Cases

| # | Query | Semantic | Faithfulness | Reason |
|---|---|---|---|---|
| 1 | What does a Senior Python Developer do? | 0.600 | 0.467 | Weak grounding in retrieved context |
| 2 | Explain the role of a Backend Engineer | 0.559 | 0.395 | Weak grounding in retrieved context |
| 3 | What skills are required for a React Frontend Developer? | 0.549 | 0.188 | Answer not grounded in retrieved context |
| 4 | What does a Full Stack Developer handle? | 0.516 | 0.167 | Answer not grounded in retrieved context |
| 5 | What does an AWS Cloud Engineer do? | 0.461 | 0.422 | Partially relevant, missing key concepts |
| 6 | What is the role of a Machine Learning Engineer? | 0.519 | 0.435 | Weak grounding in retrieved context |
| 7 | What does a Data Engineer do? | 0.432 | 0.395 | Partially relevant, missing key concepts |
| 8 | Explain NLP Engineer responsibilities | 0.826 | 0.254 | Answer not grounded in retrieved context |
| 9 | Role of Structural Engineer | 0.389 | 0.160 | Answer not grounded in retrieved context |
| 10 | What is SEO Specialist job description? | 0.495 | 0.422 | Partially relevant, missing key concepts |
| 11 | What does a Digital Marketing Specialist do? | 0.405 | 0.372 | Partially relevant, missing key concepts |
| 12 | What is the role of a Financial Analyst? | 0.450 | 0.484 | Partially relevant, missing key concepts |
| 13 | Explain Senior Accountant responsibilities | 0.775 | 0.216 | Answer not grounded in retrieved context |
| 14 | What does a Customer Success Manager do? | 0.435 | 0.400 | Partially relevant, missing key concepts |
| 15 | What does a Sales Executive do? | 0.488 | 0.269 | Answer not grounded in retrieved context |
| 16 | Explain English Teacher job role | 0.659 | 0.000 | Answer not grounded in retrieved context |
| 17 | What is Mechanical Engineer responsible for? | 0.463 | 0.115 | Answer not grounded in retrieved context |
| 18 | What does a Production Engineer do? | 0.494 | 0.400 | Partially relevant, missing key concepts |
| 19 | Explain role of DevOps Engineer | 0.471 | 0.174 | Answer not grounded in retrieved context |
| 20 | What does a Cloud Engineer do in AWS systems? | 0.569 | 0.385 | Weak grounding in retrieved context |

---

## 6. Proposed Improvements

| Improvement | Impact | Complexity |
|---|---|---|
| Switch to mistral or llama3 | Fixes language switch bug | Low |
| Use embedding-based semantic similarity | More accurate scoring | Low |
| Use larger LLM (e.g. mistral, llama3) | Reduces hallucination | Low |
| Add output length constraint in prompt | Forces context-grounded answers | Low |
| Implement RAGAS evaluation framework | Industry-standard metrics | Medium |
| Add few-shot examples to prompt | Improves answer format consistency | Low |

---

## 7. Conclusion

The generation evaluation revealed an average semantic similarity of 0.5592 and average faithfulness of 0.3905. Four distinct failure patterns were identified: LLM hallucination, semantic mismatch with ground truth, empty context responses, and non-English output in 0 queries due to the qwen2.5:3b model's Chinese language bias.

The most impactful fixes are switching to a model without Chinese bias (mistral/llama3) and implementing a cosine similarity score threshold to prevent the LLM from generating answers when retrieved context is irrelevant.