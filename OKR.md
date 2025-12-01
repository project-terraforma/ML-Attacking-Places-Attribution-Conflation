# Project Objective & Key Results (OKRs)

**Objective:** Develop rule-based and machine learning pipelines for resolving place-attribute conflation.

- **KR1 — Normalization Dataset**  
  Build and document a normalized dataset covering **20+ location patterns** that are reused by **both** the rule-based and ML pipelines, and validate that it has **<10% parsing errors**.
  - Status: Achieved. Received **0% parsing errors** and **26 locations**.

- **KR2 — Rule-Based System Accuracy**  
  Implement **5 rules per attribute** (name, phone, address, website, categories) and achieve **≥ 80% attribute-level accuracy** on the rule-based conflation dataset.
  - Status: Achieved. Received **83.16% overall accuracy**.

- **KR3 — Machine Learning Models**  
  Train **5 ML models** (one per attribute) and achieve **≥ 80% normalized, attribute-based accuracy** on the ML-based conflation dataset.
    - Status: Achieved. Received **85.60% overall accuracy**.

- **KR4 — Comparative Evaluation (Rule-Based vs ML)**  
  By Week 10, deliver evaluation comparing rule-based and ML pipelines, identifying **1–2 strengths and 1–2 limitations** considering **long-term maintainability** and **explainability**.
  - Status: Achieved.
  - Both models successfully resolve place-attribute conflation but serve different purposes in terms of **long-term maintainability** and **explainability**.

  - **Rule-Based Strengths:**
    - Explainable — every decision is traceable to explicit rules.
    - Deterministic and consistent behavior. For the same input, the rule-based logic will always produce the same output, making debugging straightforward.

  - **Rule-Based Weaknesses:**
    - Hard to scale — requires manual rule updates as new patterns arise.
    - Brittle if new patterns appear (e.g., provider changes syntax, schema, etc.), requiring manual updates.

  - **ML-Based Strengths:**
    - Learns patterns beyond what fixed rules capture.
    - Scales to large datasets without needing to manually expand rule sets.

  - **ML-Based Weaknesses:**
    - Requires larger training sets to generalize across diverse datasets.
    - Less transparent decision logic — not always clear why a particular source was chosen.
