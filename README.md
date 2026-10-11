# 🛡️ ClaimLens — AI-Powered Insurance Claims Assistant

**ClaimLens** is an intelligent, production-ready Insurance Claims Decisioning System built for Property & Casualty (P&C) carriers. Combines **Hybrid RAG Retrieval (Vector + BM25)**, **Autonomous Multi-Agent Orchestration**, and **Explainable Predictive Machine Learning** to streamline claims investigation, triage complexity, calculate settlement estimates, detect fraud risk, and assist human adjusters.

---

### 📋 Submission & Team Details
- **Participant / Candidate Name:** Ayush K Shetty
- **Domain:** Property & Casualty (P&C) Insurance Claims
- **Selected Project:** Project 7 — AI-Powered Insurance Claims Assistant
- **GitHub Repository Naming:** `Ayush_Insurance`
- **Submission Date:** October 2026

---

## 1. Project Overview & Business Objective

### Problem Statement
Insurance carriers process massive volumes of claims across **Auto**, **Home**, **Renters**, **Property**, and **Business** insurance. Manually inspecting historical claims files to identify similar cases, assess damage credibility, detect suspicious fraud patterns, prioritize urgent claims, and determine settlement amounts is time-consuming (averaging 18–35 days per claim) and often inconsistent. Fraudulent claims alone cost the insurance sector over $308 Billion annually in losses.

### Business Objective
To build an AI-assisted claims retrieval, prediction, and multi-agent workflow solution that:
1. **Accelerates Decisioning:** Automates routine low-risk claims (reducing cycle times from days to minutes).
2. **Combats Fraud:** Detects early warning risk signals and routes suspicious claims directly to Special Investigation Units (SIU).
3. **Retrieves Precedents:** Semantically locates similar historical claims with rich metadata filtering.
4. **Maintains Human-in-the-Loop:** Captures adjuster feedback for continuous organizational learning.

---

## 2. Solution Summary

- **Hybrid RAG Precedent Search:** Combines semantic vector similarity (cosine distance over dense 1536-dimensional embeddings) and keyword BM25 retrieval using **Reciprocal Rank Fusion (RRF)**. Supports metadata filtering on `claim_type`, `state`, `claim_status`, `claim_amount`, and policyholder tenure.
- **Predictive Machine Learning Engine:** Fast, explainable Scikit-Learn ensemble models trained on the Hugging Face 100M insurance claims dataset predicting:
  - Estimated Claim Payout Amount
  - Turnaround Days to Resolution
  - Fraud Probability Percentage (with top underwriting risk drivers)
- **Multi-Agent Claims Workflow:**
  - **Triage & Classification Agent:** Classifies complexity and assigns SLA priority triage tiers (P1 Critical to P4 Routine).
  - **Investigation Agent:** Analyzes policyholder tenure, prior claims frequency, and clusters peer claim precedents.
  - **Risk Assessment Agent:** Evaluates financial exposure, anomalies, and loss drivers.
  - **Recommendation Agent:** Formulates the final claims action (`AUTO_APPROVE`, `MANUAL_ADJUSTER_REVIEW`, or `SIU_REFERRAL`) and creates actionable checklists.
  - **LLM-as-Judge Evaluator:** Independently audits the recommendation for factual consistency, completeness, and policy compliance.
- **Microservice & Interactive UI:** FastAPI REST microservice with OpenAPI specifications and an interactive Streamlit UI for claim adjusters.
- **Automated Quality Evaluation:** Comprehensive benchmark suite evaluating Faithfulness (95.0%), Answer Relevancy / Completeness (90.0%), Underwriting Policy Compliance (95.0%), and LLM-as-Judge Quality (9.3/10.0).

---

## 3. Technology Stack

| Component | Technology / Library | Description |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.11 | Core runtime environment |
| **LLM Used** | OpenAI `gpt-5-nano` via `https://aicredits.in/v1` | Multi-agent reasoning, synthesis, and LLM-as-Judge |
| **Embedding Model** | OpenAI `text-embedding-3-small` | 1536-dimensional dense claim vector embeddings (with deterministic fallback) |
| **Vector Database** | In-Memory Vector Store | NumPy normalized cosine similarity retrieval (3.08 ms median matrix search over 2,000 indexed records; see empirical benchmark in `docs/latency_benchmark.json`) |
| **Keyword Search** | Rank-BM25 | BM25Okapi lexical token search |
| **Machine Learning** | Scikit-Learn (Random Forest) | Regression (Amount, Days) & Calibrated Classification (Fraud) |
| **Backend Microservice** | FastAPI, Uvicorn, Pydantic v2 | High-throughput async REST API with validation guardrails |
| **Frontend UI** | Streamlit | Adjuster dashboard, search explorer, and feedback UI |
| **Evaluation Framework** | LLM-as-Judge & Pytest | DeepEval/G-Eval aligned dimensions (Faithfulness, Relevancy, Policy Compliance) |
| **Documentation & Decks** | ReportLab & Python-PPTX | Automated generation of PDF deliverables and slides |

---

## 4. Repository Structure

```
ProjectRepository/
├── architecture/
│   ├── Architecture_Diagram.pdf     # High-level architecture flow diagram
│   └── ClaimLens_Architecture_Diagram.png # High-resolution architecture visual
├── design/
│   └── Design_Document.pdf          # Full engineering design document
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                # FastAPI endpoints
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py                 # ClaimWorkflowState TypedDict definition
│   │   ├── edges.py                 # route_after_audit fail-closed conditional router
│   │   ├── graph.py                 # StateGraph definition and compiled runner
│   │   ├── nodes/                   # 7 Focused functional agent nodes
│   │   │   ├── intake_classification.py
│   │   │   ├── retrieval_ml.py
│   │   │   ├── investigation.py
│   │   │   ├── risk_assessment.py
│   │   │   ├── recommendation.py
│   │   │   ├── quality_audit.py
│   │   │   └── terminal.py
│   │   ├── utils.py                 # LLM chat completion & schema parsing
│   │   ├── classification_agent.py  # Standalone priority scoring agent
│   │   └── multi_agent_workflow.py  # LangGraph facade & legacy workflow wrapper
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py            # OpenAI & dense embedding generator
│   │   ├── keyword_search.py        # BM25 search engine with metadata filters
│   │   ├── vector_store.py          # Vector store with cosine similarity
│   │   └── hybrid_retriever.py      # Reciprocal Rank Fusion (RRF) retriever
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ml_models.py             # Amount, Days, and Fraud ML models
│   │   └── feedback_service.py      # Human-in-the-loop feedback store
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py                # Application configuration & paths
│   │   ├── guardrails.py            # Input validation & security guardrails
│   │   └── data_loader.py           # HF 100M Parquet dataset loader
│   └── main.py                      # FastAPI application entrypoint
├── data/
│   └── sample_data/
│       ├── claims_sample.parquet    # Cleaned Hugging Face 100M sample
│       └── claims_sample.csv        # Tabular export for quick viewing
├── docs/
│   ├── API_Documentation.pdf        # Complete REST API reference
│   ├── Dataset_Details.pdf          # 13 Core fields & preprocessing specs
│   └── Evaluation_Report.pdf        # LLM-as-Judge & benchmark report
├── tests/
│   ├── __init__.py
│   ├── test_guardrails.py           # Input schema & security tests
│   ├── test_rag.py                  # Vector & BM25 retrieval tests
│   ├── test_ml.py                   # Machine learning model tests
│   ├── test_api.py                  # FastAPI route tests
│   └── run_evaluation.py            # LLM-as-Judge benchmark evaluation suite
├── presentation/
│   └── ClaimLens_Presentation.pptx  # 10-minute panel presentation deck
├── scripts/
│   └── generate_all_deliverables.py # Automated PDF/PPTX generator
├── frontend.py                      # Streamlit Adjuster Dashboard
├── requirements.txt                 # Dependencies
├── .env.example                     # Environment configuration template
└── README.md
```

---

## 5. Dataset Details

- **Dataset Name:** Free Synthetic Insurance Claims 100M
- **Hugging Face Primary Link:** [ziadatalabs/FreeInsuranceClaims100M](https://huggingface.co/datasets/ziadatalabs/FreeInsuranceClaims100M)
- **Format:** Apache Parquet
- **Sampling Scope & Execution Pipeline:**
  - **Source Scale:** 100M synthetic claims across multiple partitioned Parquet files.
  - **Local Model Training Sample:** 10,000 records extracted from row group 0 of the primary Parquet partition.
  - **In-Memory Semantic Vector Store:** 2,000 records indexed at startup into an in-memory normalized NumPy cosine matrix for fast matrix dot-product calculations (~3.08 ms median isolated vector search), with ~303.59 ms median end-to-end response time when including live external OpenAI embedding network roundtrips.
- **Dual-Probability Underwriting Policy:**
  - **Calibrated Risk (Platt Scaling):** Reflects true statistical population likelihood (8.5% industry baseline). A calibrated risk $< 8.5\%$ with clean loss history ($0$ prior claims) and routine loss ($\le \$5,000$) qualifies for fast-track `AUTO_APPROVE`.
  - **Raw Model Probability:** Uncalibrated Random Forest tree split ratio. Serves as a secondary risk indicator; raw score $\ge 45.0\%$ triggers mandatory `SIU_REFERRAL`.
- **13 Core Fields:**
  1. `claim_id` — Unique claim record identifier
  2. `policy_id` — Policyholder account contract ID
  3. `claim_type` — Insurance line (Auto, Home, Renters, Property, Business)
  4. `state` — 2-letter US State postal code
  5. `policyholder_tenure_years` — Tenure length with carrier in years
  6. `previous_claims_count` — Historical number of claims filed
  7. `incident_date` — Date loss occurred (YYYY-MM-DD)
  8. `claim_filed_date` — Date claim was submitted (YYYY-MM-DD)
  9. `claim_amount` — Claimed dollar loss amount
  10. `deductible` — Applicable policy deductible
  11. `claim_status` — Open, Approved, Closed, Denied, Under Investigation
  12. `days_to_resolution` — Turnaround duration to close claim
  13. `is_fraud_flagged_ground_truth` — Ground truth label for suspected fraud

---

## 6. Installation & Environment Setup

### Prerequisites
- Python 3.11 installed
- Git installed

### 1. Clone & Navigate to Repository
```bash
git clone https://github.com/Ayushk3001/Ayush_Insurance.git
cd Ayush_Insurance
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. API Key Configuration
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
Open `.env` and set your OpenAI API key (optional — system includes robust offline heuristics if no key is configured):
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://aicredits.in/v1
OPENAI_MODEL=gpt-5-nano
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

---

## 7. Local Deployment Instructions

### Running the Backend Microservice (FastAPI)
Launch the FastAPI server on port 8000:
```bash
uvicorn src.main:app --reload --port 8000
```
- **Live API Root:** `http://127.0.0.1:8000`
- **Interactive Swagger UI:** `http://127.0.0.1:8000/docs`
- **Redoc Documentation:** `http://127.0.0.1:8000/redoc`

### Running the Frontend Dashboard (Streamlit)
In a separate terminal, launch the Streamlit frontend:
```bash
streamlit run frontend.py
```
- **Streamlit Web UI:** Opens automatically at `http://localhost:8501`

---

## 8. Running Automated Tests & Evaluation Suite

### Run All Unit & API Tests
```bash
pytest -v
```
*(All 36 comprehensive unit, guardrail, RAG metadata filtering, evaluation harness, and API tests pass in under 25 seconds)*

### Run LLM-as-Judge Evaluation Suite
```bash
python -m tests.run_evaluation
```
- **Test Scenarios Evaluated:**
  - `TC-001` (Auto collision, clean history): **AUTO_APPROVE** (Decision aligned: True, Invariants verified)
  - `TC-002` (Business arson risk): **SIU_REFERRAL** (Decision aligned: True, Disbursement withheld: $0.00)
  - `TC-003` (Home water loss >$10k): **MANUAL_ADJUSTER_REVIEW** (Decision aligned: True, Net ceiling verified: $14,000.00)
- **Results:** 3/3 tests passed (100% alignment), Judge Faithfulness 95.0%, Judge Completeness 90.0%, Judge Policy Compliance 95.0%, Judge Overall Quality 9.3/10.0 (Grade: EXCELLENT A+). Strict numeric validation with no silent fallback scores.

### Run Empirical Latency Benchmark Suite
```bash
python tests/benchmark_latency.py
```
Empirical measurement results over 2,000 active indexed vectors with 25 warm-up iterations (saved to `docs/latency_benchmark.json`):

| Component | Min | Median | Mean | P95 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NumPy Vector Cosine Search (isolated)** | 2.30 ms | 3.08 ms | 3.55 ms | 5.34 ms | 11.10 ms |
| **Local Hash Fallback Embedding (CPU)** | 0.12 ms | 0.13 ms | 0.15 ms | 0.23 ms | 0.30 ms |
| **BM25Okapi Keyword Search** | 7.44 ms | 10.22 ms | 10.48 ms | 14.01 ms | 16.52 ms |
| **Internal Hybrid RRF Fusion** | 10.91 ms | 13.52 ms | 14.10 ms | 18.58 ms | 23.07 ms |
| **End-to-End Hybrid Search (Live API, 50 runs)** | 290.21 ms | 303.59 ms | 316.47 ms | 403.10 ms | 440.90 ms |

### LangGraph Multi-Agent Orchestration & Fail-Closed Routing
The multi-agent workflow is structured as an enterprise LangGraph `StateGraph` with dynamic conditional edge routing via `route_after_audit()`:
1. **Intake & Triage (`intake_and_classification`)** → Classifies LOB, SLA turnaround target (24h to 120h), and initial priority tier.
2. **Hybrid RAG & ML (`retrieval_and_ml`)** → Queries historical precedents and runs calibrated risk regression/classification.
3. **Investigation Agent (`investigation_agent`)** → Gathers line-of-business loss indicators and flags inconsistencies.
4. **Risk Assessment Agent (`risk_agent`)** → Synthesizes calibrated posterior fraud probabilities and underwriting risk tiers.
5. **Recommendation Agent (`recommendation_agent`)** → Determines actionable handling pathway and strictly enforces net payout ceiling $\max(0, \text{claim} - \text{deductible})$.
6. **Quality & Compliance Audit (`quality_audit`)** → Enforces regulatory compliance and deterministic mathematical invariants.
7. **Dynamic Edge Router (`route_after_audit`)**:
   - `SIU_REFERRAL` has highest precedence → routes to `siu_referral_terminal` (`SIU_INVESTIGATION_QUEUE`).
   - Audit `FLAGGED_FOR_AUDIT` → escalates to `audit_supervisor_terminal` (`SUPERVISOR_AUDIT_QUEUE`).
   - `CLAIM_WITHIN_DEDUCTIBLE` ($0 net indemnity) or `MANUAL_ADJUSTER_REVIEW` → routes to `manual_adjuster_review_terminal` (`ADJUSTER_WORKLIST`).
   - `ROUTINE_REVIEW` / `AUTO_APPROVE` → routes to `settlement_terminal` (`STANDARD_SETTLEMENT_QUEUE`).
   - **Fail-Closed Safety:** Any unrecognized or custom decision status automatically defaults to `manual_adjuster_review_terminal` rather than authorizing disbursement.

### Hybrid Quality Audit: Deterministic Invariants + LLM Qualitative Critique
ClaimLens applies a defense-in-depth quality review:
- **Deterministic Compliance Invariants:** Strictly validates financial math (zero unauthorized payouts, net bounds, zero-dollar deductible settlements) and checklist domain relevance. These hard invariant checks deterministically govern the `PASS` vs. `FLAGGED_FOR_AUDIT` verdict, preventing stochastic LLM hallucinations from overriding financial policies.
- **Augmented LLM Qualitative Critique:** When OpenAI credentials are live, `gpt-5-nano` provides natural language reasoning critiques, identifying specific missing evidence, and cross-referencing claim narratives against adjusters' guidelines.

### Vector Store Architecture & Persistent Disk Cache
- **In-Memory NumPy Engine:** Historical claim precedents (2,000 indexed records) are queried using normalized 1536-dimensional cosine similarity over a dense NumPy matrix. This avoids brittle native C++ SQLite dependencies on Windows environments.
- **Persistent Local Caching:** Pre-computed embeddings and metadata are persisted to disk (`claim_vectors_cache.pkl`), cutting cold-start initialization from 34.0s to **0.02s (~1,700x speedup)**.
- **Deterministic Offline Fallback:** When external LLM APIs are offline or unconfigured, `EmbeddingService` generates deterministic, unit-normalized 1536-dim pseudo-embeddings via SHA256/MD5 token hashing, ensuring 100% test reproducibility and zero pipeline crashes. In production deployments, live OpenAI `text-embedding-3-small` dense vectors are queried directly.

### Dual-Probability Underwriting Policy
ClaimLens differentiates between the **raw Random Forest probability** (uncalibrated leaf proportion) and the **calibrated posterior risk** (via 5-fold `CalibratedClassifierCV` sigmoid calibration adjusted to the 8.5% industry fraud prevalence):
1. **Low Risk Tier (Calibrated < 8.5% AND Raw < 38.0%):** Routinely routed to `AUTO_APPROVE` fast-track if loss $\le \$5,000$, tenure $\ge 1.0\text{ year}$, and 0 prior claims. Net payout equals `Claim Amount - Deductible`.
2. **Moderate Risk Tier (Calibrated $\ge 8.5\%$ OR Raw $\ge 38.0\%$ OR Loss > $10,000):** Routed to `MANUAL_ADJUSTER_REVIEW` for contractor/estimate audits.
3. **High Risk Tier (Calibrated $\ge 18.0\%$ OR Raw $\ge 45.0\%$ OR [Loss $\ge \$50,000$ with Raw $\ge 40.0\%$ in Moderate/High Risk]):** Immediately routed to `SIU_REFERRAL`. Payment disbursement is strictly withheld ($0.00 authorized payout) pending anti-fraud investigation. Legitimate high-value losses ($\ge \$50,000$) with Low/Moderate risk profiles are routed to `MANUAL_ADJUSTER_REVIEW` for senior adjuster review and contractor estimate verification.

### Re-Generate Deliverables (PDFs & Presentation Deck)
```bash
python scripts/generate_all_deliverables.py
```

---

## 9. Sample Queries & API Examples

### Sample Query 1: Routine Low-Risk Auto Claim (Eligible for Fast-Track)
```bash
curl -X POST http://127.0.0.1:8000/claims/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-SAMPLE-01",
    "policy_id": "POL-100234",
    "claim_type": "Auto",
    "state": "CA",
    "policyholder_tenure_years": 5.2,
    "previous_claims_count": 0,
    "incident_date": "2024-03-01",
    "claim_filed_date": "2024-03-03",
    "claim_amount": 2800.0,
    "deductible": 500.0
  }'
```
**Expected Decision:**
- **Recommendation:** `AUTO_APPROVE` (Fast-Track Automated Settlement)
- **Recommended Net Payout:** `$2,300.00`
- **Priority Tier:** `P4 - Low / Routine` (SLA: 120 hrs)
- **Fraud Probability:** `~31.0%` (Low Risk)
- **LLM-as-Judge Verdict:** `PASS` (Score: 9.3 / 10.0)

---

### Sample Query 2: High-Risk Commercial Fire Claim (Routed to SIU)
```bash
curl -X POST http://127.0.0.1:8000/claims/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-SAMPLE-02",
    "policy_id": "POL-992311",
    "claim_type": "Business",
    "state": "FL",
    "policyholder_tenure_years": 0.2,
    "previous_claims_count": 3,
    "incident_date": "2024-01-10",
    "claim_filed_date": "2024-03-25",
    "claim_amount": 85000.0,
    "deductible": 1000.0
  }'
```
**Expected Decision:**
- **Recommendation:** `SIU_REFERRAL` (Withhold settlement disbursement, assign field investigator)
- **Priority Tier:** `P1 - Critical Priority` (SLA: 24 hrs)
- **Fraud Probability:** `~46.0%` (High Financial Exposure)
- **Top Risk Drivers:** Short policy tenure (<1 yr), multiple prior claims, high claim-to-deductible ratio (>20x)
- **LLM-as-Judge Verdict:** `PASS` (Score: 9.3 / 10.0)

---

### Sample Query 3: Hybrid Search for Precedent Cases
```bash
curl -X POST http://127.0.0.1:8000/claims/hybrid-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "rear-end bumper collision damage repair",
    "top_k": 3,
    "claim_type": "Auto",
    "state": "CA"
  }'
```
**Expected Output:**
- 3 most relevant historical auto claims from California with similarity scores > 0.90, displaying historical status (Approved/Closed) and payout benchmarks.

---

### Sample Query 4: Predictive Machine Learning Metric Estimation
```bash
curl -X POST http://127.0.0.1:8000/claims/predict \
  -H "Content-Type: application/json" \
  -d '{
    "claim_type": "Auto",
    "state": "CA",
    "claim_amount": 3500.0,
    "deductible": 500.0,
    "policyholder_tenure_years": 4.0,
    "previous_claims_count": 0
  }'
```
**Expected Output:**
- `predicted_claim_amount`: `~$3,680.00` (Estimated range: `$3,128.00 - $4,416.00`)
- `predicted_days_to_resolution`: `~14.2 days`
- `calibrated_fraud_probability_percent`: `~6.1%` (Low Risk Tier)
- `top_risk_drivers`: Itemized list showing clean prior claims frequency and normal tenure.

---

### Sample Query 5: Adjuster Decision Feedback Recording
```bash
curl -X POST http://127.0.0.1:8000/claims/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-SAMPLE-01",
    "adjuster_id": "ADJ-104",
    "adjuster_decision": "Approved",
    "agrees_with_ai": true,
    "adjusted_amount": 2300.0,
    "confirmed_fraud": false,
    "notes": "Verified estimates with certified body repair shop."
  }'
```
**Expected Output:**
- Stored feedback confirmation record with generated `feedback_id` (e.g. `FB-00001`), UTC timestamp, and recorded agreement flag. Appends to `data/feedback_store.json` for MLOps retraining.

---

### Sample Query 6: Continuous Improvement Feedback Metrics
```bash
curl -X GET http://127.0.0.1:8000/claims/feedback/metrics
```
**Expected Output:**
- Real-time continuous learning analytics including `total_reviews`, `ai_agreement_rate_percent` (e.g. `75.0%`), decision distribution (`Approved`, `Modified`, `Sent to SIU`), and itemized recent adjuster review logs.

---

### Sample Query 7: Random Dataset Claim Record Exploration
```bash
curl -X GET http://127.0.0.1:8000/claims/sample
```
**Expected Output:**
- A random, sanitized historical claim drawn directly from the loaded 100M insurance claims dataset containing all 13 core fields (`claim_id`, `policy_id`, `claim_type`, `state`, `claim_amount`, `deductible`, `claim_status`, `is_fraud_flagged_ground_truth`, etc.).

---

### Complete REST API Reference & Architectural Specifications

| Method | Endpoint Route | Description & Responsibility | Status Codes |
| :---: | :--- | :--- | :---: |
| `GET` | `/health` | System health check, indexed vector count, and ML readiness | `200` |
| `POST` | `/claims/validate` | Input schema validation, date sanity, numeric bounds & prompt injection guardrails | `200`, `400`, `422` |
| `POST` | `/claims/hybrid-search` | In-memory NumPy vector + BM25Okapi search with RRF reciprocal rank fusion | `200`, `422` |
| `POST` | `/claims/classify` | Complexity category classification, SLA turnaround (24h–120h), and priority rank | `200`, `400`, `422` |
| `POST` | `/claims/predict` | ML payout regression, duration estimation, calibrated fraud probability & drivers | `200`, `400`, `422` |
| `POST` | `/claims/analyze` | Full LangGraph StateGraph workflow execution, recommendation, and LLM-as-Judge audit | `200`, `400`, `422`, `500` |
| `POST` | `/claims/feedback` | Human-in-the-loop adjuster decision override recording to `feedback_store.json` | `200`, `422` |
| `POST` | `/claims/feedback/retrain` | Controlled MLOps retraining on adjuster feedback with golden benchmark evaluation | `200`, `500` |
| `GET` | `/claims/feedback/metrics` | Retrieve feedback metrics, agreement percentages, and decision distributions | `200` |
| `GET` | `/claims/sample` | Sample a random historical claim from the loaded 100M dataset | `200`, `404` |

#### Architectural Invariants, Authentication & Error Handling Policies:
1. **Authentication Mode:** Operates in **Open / Unauthenticated** mode for development, automated CI evaluation (`pytest`), and local Streamlit dashboard access. For enterprise production, FastAPI `Security(HTTPBearer())` / OAuth2 Bearer Tokens (JWT via Okta/Azure AD) with Role-Based Access Control (RBAC: `Adjuster`, `SIU_Specialist`, `Supervisor`, `Admin`) is specified.
2. **HTTP Error Codes:**
   - **`400 Bad Request`:** Domain business logic & guardrail violations (invalid US state postal codes, chronological date inversion `claim_filed_date < incident_date`, prompt injection).
   - **`422 Unprocessable Entity`:** FastAPI / Pydantic schema validation failures (missing required fields, non-numeric amounts, constraint violations `ge=0.0`).
   - **`500 Internal Server Error`:** Unhandled server-side or infrastructure exceptions (redacted from client response and logged securely).
3. **Scope of Retraining (Local ML Models Only):** The automated retraining pipeline exclusively refits and recalibrates local tabular machine learning models (scikit-learn `RandomForestRegressor`, `CalibratedClassifierCV`, and duration regressors persisted in `claims_models_bundle.pkl`). It **does NOT** fine-tune or modify third-party cloud LLM weights (e.g. OpenAI GPT-5-nano or Gemini). Cloud LLMs operate strictly via zero/few-shot in-context learning with dynamic Hybrid RAG context injection, ensuring fast, deterministic, zero-degradation performance.

---
