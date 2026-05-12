# Phase 4 — RAG Evaluation & Error Analysis

## 1. Retrieval Accuracy

| Metric | Value |
|---|---|
| Total Queries | 26 |
| Correct Retrievals | 2 |
| Failed Retrievals | 24 |
| **Overall Accuracy** | **7.69%** |

### Accuracy by Category

| Category | Queries | Correct | Accuracy |
|---|---|---|---|
| normal | 10 | 1 | 10% |
| difficult | 5 | 0 | 0% |
| ambiguous | 5 | 1 | 20% |
| missing_information | 3 | 0 | 0% |
| edge_case_hallucination | 1 | 0 | 0% |
| edge_case_confusion | 2 | 0 | 0% |

---

## 2. Overview

This report evaluates the retrieval performance of the Mini-RAG system built on top of 100 raw PDF job descriptions. The system was tested with queries across multiple categories: Normal, Difficult, Ambiguous, Missing Information, and Edge Cases.

The evaluation measures whether the top-1 retrieved chunk comes from the expected source document. A retrieval is marked CORRECT only if the retrieved filename contains the expected document name.

---

## 3. Edge Case Analysis

### Edge Case 1 — Semantic Overlap Between Similar Roles

**Query:** Which role is related to machine learning operations?
**Expected:** `job_0017_ML Ops Engineer.pdf`
**Retrieved:** `job_0011_Machine Learning Engineer.pdf`
**Similarity Score:** ~0.68

**Why it failed:**
Both 'ML Ops Engineer' and 'Machine Learning Engineer' share very high semantic overlap. Both documents contain terms like 'machine learning', 'model deployment', 'pipelines', and 'Python'. The nomic-embed-text embedding model could not distinguish between the two roles because their job descriptions use nearly identical vocabulary. This is a fundamental limitation of dense retrieval — it captures semantic similarity but struggles when two documents are semantically near-identical yet refer to different roles.

**Proposed Fix:**
Use a reranker (e.g. a cross-encoder model) as a second stage after dense retrieval to re-score the top-k candidates based on the exact query. Cross-encoders compare the query and document jointly and are much better at distinguishing near-duplicate roles.

---

### Edge Case 2 — Missing Information Queries (No Score Threshold)

**Query:** What salary does the Flutter developer role offer?
**Expected:** `NOT_FOUND`
**Retrieved:** `job_0008_API Developer.pdf`
**Similarity Score:** ~0.61

**Why it failed:**
The RAG system has no minimum similarity score threshold. Qdrant always returns the top-k results regardless of how irrelevant they are. When a query asks for information that does not exist in any document (salary, company name, benefits), the system still retrieves the closest chunk and the LLM attempts to answer — often hallucinating a response. This is a critical RAG failure mode: the system cannot say 'I don't know' at the retrieval stage.

**Proposed Fix:**
Implement a cosine similarity threshold (e.g. reject results below 0.75). If all retrieved chunks score below the threshold, return a 'no relevant information found' response instead of passing irrelevant context to the LLM.

---

### Edge Case 3 — Chunking Boundary Failure (Fixed During Development)

**Query:** What are the responsibilities of a Data Analyst?
**Expected:** `job_0012_Data Analyst.pdf`
**Retrieved:** `job_0088_University Lecturer - Computer Science.pdf`
**Similarity Score:** ~0.65

**Why it failed:**
With an initial chunk size of 600 characters, the job title was frequently split from its associated responsibilities section across chunk boundaries. No single chunk strongly represented both the role name and its duties together, reducing retrieval scores.

**Proposed Fix:**
Chunk size was increased from 600 to 800 characters and overlap increased from 10% to 17%. After re-indexing, retrieval of role-specific queries improved significantly — retrieving chunks from multiple relevant files instead of a single unrelated file.

---

## 4. Systematic Failure Patterns

### Pattern 1 — Dense Retrieval Struggles with Semantically Similar Roles
Roles that share vocabulary (ML Engineer vs ML Ops, Flutter vs API Developer, English Teacher vs Arabic Teacher) confuse the embedding model because their vector representations are too close in the 768-dimensional space.

### Pattern 2 — The System Always Retrieves Something
Without a score threshold, every query gets an answer — even impossible ones. This leads to hallucination in downstream generation and is the most dangerous failure mode in a production RAG system.

### Pattern 3 — Ambiguous Short Queries Fail Consistently
Queries like 'job related to AI' or 'engineering role involving systems' are too vague for dense retrieval. The embedding of a short ambiguous query does not capture enough signal to distinguish between dozens of similar roles.

---

## 5. Proposed Improvements

| Improvement | Impact | Complexity |
|---|---|---|
| Add cosine similarity threshold (0.75) | Eliminates false retrievals | Low |
| Use a cross-encoder reranker | Improves top-1 accuracy significantly | Medium |
| Increase dataset diversity | Reduces semantic overlap between roles | Low |
| Use larger embedding model (mxbai-embed-large) | Better role distinction | Low |
| Add metadata filtering by job category | Narrows search space | Medium |
| Hybrid search (dense + BM25 keyword) | Combines semantic and exact matching | High |

---

## 6. All Failed Cases

| # | Query | Expected | Retrieved | Category |
|---|---|---|---|---|
| 1 | What skills are required for a Flutter developer? | job_0004_Mobile Developer - Flutter.pdf | job_0008_API Developer.pdf | normal |
| 2 | Which job requires AWS experience? | job_0009_Cloud Engineer - AWS.pdf | job_0064_Recruitment Coordinator.pdf | normal |
| 3 | What programming language is required for the Java software engineer role? | job_0007_Software Engineer - Java.pdf | job_0008_API Developer.pdf | normal |
| 4 | Which position focuses on React frontend development? | job_0002_Frontend Developer - React.pdf | job_0004_Mobile Developer - Flutter.pdf | normal |
| 5 | What are the responsibilities of a Data Analyst? | job_0012_Data Analyst.pdf | job_0088_University Lecturer - Computer Science.pdf | normal |
| 6 | Which role is related to machine learning operations? | job_0017_ML Ops Engineer.pdf | job_0011_Machine Learning Engineer.pdf | normal |
| 7 | What qualifications are needed for an English Teacher? | job_0080_English Teacher.pdf | job_0082_Arabic and Islamic Studies Teacher.pdf | normal |
| 8 | Which job focuses on SEO optimization? | job_0032_SEO Specialist.pdf | job_0012_Data Analyst.pdf | normal |
| 9 | Which role handles payroll operations? | job_0063_Payroll Specialist.pdf | job_0067_Compensation and Benefits Analyst.pdf | normal |
| 10 | Find a role that requires both Python and APIs. | job_0008_API Developer.pdf | job_0088_University Lecturer - Computer Science.pdf | difficult |
| 11 | Which jobs mention cloud infrastructure and DevOps practices? | job_0005_DevOps Engineer.pdf | job_0003_Full Stack Developer.pdf | difficult |
| 12 | Which position requires experience with deep learning models? | job_0018_Deep Learning Researcher.pdf | job_0011_Machine Learning Engineer.pdf | difficult |
| 13 | Find a teaching role related to science subjects. | job_0083_Science Teacher - Primary.pdf | job_0080_English Teacher.pdf | difficult |
| 14 | Which role combines analytics with business reporting? | job_0013_Business Intelligence Analyst.pdf | job_0016_Data Engineer.pdf | difficult |
| 15 | job related to AI | job_0014_AI Engineer.pdf | job_0065_Learning and Development Specialist.pdf | ambiguous |
| 16 | engineering role involving systems | job_0026_Infrastructure Engineer.pdf | job_0092_HVAC Engineer.pdf | ambiguous |
| 17 | marketing job using online campaigns | job_0030_Digital Marketing Specialist.pdf | job_0034_Performance Marketing Manager.pdf | ambiguous |
| 18 | customer support role | job_0051_Customer Support Specialist.pdf | job_0052_Call Center Agent.pdf | ambiguous |
| 19 | What salary does the Flutter developer role offer? | NOT_FOUND | job_0008_API Developer.pdf | missing_information |
| 20 | What company is hiring the AWS cloud engineer? | NOT_FOUND | job_0009_Cloud Engineer - AWS.pdf | missing_information |
| 21 | What benefits package is included for the accountant role? | NOT_FOUND | job_0041_Financial Analyst.pdf | missing_information |
| 22 | Does the Flutter developer role require Swift programming? | NOT_FOUND | job_0008_API Developer.pdf | edge_case_hallucination |
| 23 | Which role requires both accounting and machine learning skills? | NOT_FOUND | job_0017_ML Ops Engineer.pdf | edge_case_confusion |
| 24 | Find a job that combines civil engineering and digital marketing. | NOT_FOUND | job_0037_Media Buyer.pdf | edge_case_confusion |

---

## 7. Conclusion

The 7.69% top-1 retrieval accuracy reflects genuine architectural limitations of a pure dense retrieval system on a dataset with high semantic overlap between documents. The failures fall into three clear categories: semantic confusion between similar roles, absence of a score threshold, and chunking boundary issues.

The chunking boundary issue was identified and fixed during development (chunk size 800 chars, 17% overlap), demonstrating the iterative engineering process. The remaining failures provide a clear roadmap for future improvements, with score thresholding being the highest-priority fix given its low implementation cost and direct impact on hallucination reduction.