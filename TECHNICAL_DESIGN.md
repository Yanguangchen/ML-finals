# Fraud Detection Project - Technical Design Notes

## 1) Purpose

This document adds implementation-focused technical structure to the notebook workflow.
It is intended to make future changes safer, faster, and easier to review.

## 2) When ER diagrams are appropriate here

- **Not appropriate for the current source dataset itself**:
  - the training file is a single anonymized tabular dataset (PCA features V1-V28 + Time + Amount + Class),
  - there is no normalized relational schema in the project repository.
- **Appropriate for production design**:
  - a conceptual ER model is useful for fraud operations systems (transactions, predictions, alerts, analyst feedback, retraining metadata).

So, this doc includes a **conceptual production ER diagram**, not a physical schema claim about the current CSV.

---

## 3) Offline training/evaluation pipeline (implemented in notebook)

```mermaid
flowchart TD
    A[Raw transactions dataset] --> B[Time-aware split: train / val / test]
    B --> C[Fit scaler on train only]
    C --> D[Transform train / val / test]
    D --> E[Baselines: trivial + logistic regression]
    D --> F[Starting MLP]
    F --> G[Threshold search on validation set]
    F --> H[Scale-up overfit check]
    H --> I[Regularize + hyperparameter experiments]
    I --> J[Multi-seed stability on validation]
    J --> K[Fixed-seed handoff model and threshold]
    K --> L[Retrain on train+val using selected epoch]
    L --> M[Official test report]
    M --> N[Calibration + error analysis]
    N --> O[Optional post-hoc seed robustness on test]
```

### Protocol guardrails

1. Validation is used for threshold/hyperparameter decisions.
2. Test is not used for selection.
3. Optional multi-seed test robustness is post-hoc sensitivity analysis only.

---

## 4) Real-time serving flow (deployment-oriented)

```mermaid
sequenceDiagram
    participant P as Payment Gateway
    participant FS as Feature Service
    participant MS as Model Service
    participant RE as Rules Engine
    participant AR as Analyst Review Queue
    participant FB as Feedback Store

    P->>FS: Transaction event
    FS->>MS: Feature vector for scoring
    MS-->>RE: Fraud probability + model_version
    RE-->>P: Decision (approve / step-up / block / review)
    RE->>AR: Create review case (if medium/high risk)
    AR-->>FB: Analyst label (fraud / legit)
    FB-->>MS: Labeled outcomes for periodic retraining
```

---

## 5) Conceptual production ER diagram (optional, but appropriate)

```mermaid
erDiagram
    CARDHOLDER ||--o{ TRANSACTION : initiates
    MERCHANT ||--o{ TRANSACTION : receives
    TRANSACTION ||--|| PREDICTION : scored_as
    MODEL_VERSION ||--o{ PREDICTION : produces
    TRANSACTION ||--o| ALERT : may_trigger
    ALERT ||--o| ANALYST_REVIEW : reviewed_as
    TRANSACTION ||--o{ LABEL_FEEDBACK : eventually_labeled
    MODEL_VERSION ||--o{ RETRAIN_RUN : created_by

    CARDHOLDER {
      string cardholder_id
      string risk_segment
    }

    MERCHANT {
      string merchant_id
      string mcc
      string region
    }

    TRANSACTION {
      string tx_id
      datetime tx_time
      decimal amount
      string cardholder_id
      string merchant_id
    }

    MODEL_VERSION {
      string model_version_id
      datetime deployed_at
      float threshold_high
      float threshold_medium
    }

    PREDICTION {
      string tx_id
      string model_version_id
      float fraud_prob
      string decision
      datetime scored_at
    }

    ALERT {
      string alert_id
      string tx_id
      string risk_tier
      string status
    }

    ANALYST_REVIEW {
      string review_id
      string alert_id
      string outcome
      datetime reviewed_at
    }

    LABEL_FEEDBACK {
      string tx_id
      int label
      datetime label_time
      string source
    }

    RETRAIN_RUN {
      string retrain_run_id
      string model_version_id
      datetime started_at
      datetime completed_at
      string data_window
    }
```

---

## 6) Data contracts (minimal)

### Scoring input contract

| Field | Type | Notes |
|---|---|---|
| tx_id | string | Unique transaction identifier |
| tx_time | datetime | Event time for windowing and drift checks |
| features | vector<float> | Aligned with model training feature order |
| model_version_id | string | Explicit version pinning |

### Scoring output contract

| Field | Type | Notes |
|---|---|---|
| tx_id | string | Echo input key |
| fraud_prob | float | 0..1 |
| decision | enum | approve / step_up / review / block |
| threshold_snapshot | object | Thresholds used at score time |
| scored_at | datetime | Auditability |

---

## 7) Monitoring and retraining triggers

Track at minimum:

- Precision, recall, PR AUC by rolling window
- Alert volume and analyst acceptance rate
- Probability distribution shift (PSI/KL or similar)
- Data freshness and feature null-rate checks

Trigger retraining when:

1. PR AUC or recall drops below agreed floor for N windows, or
2. Distribution shift exceeds threshold, or
3. Business policy/threshold updates require recalibration.

---

## 8) How this document should be used

- Keep notebook narrative for pedagogy.
- Keep this file for system-level architecture, contracts, and operations.
- Update this file whenever threshold policy, retraining protocol, or deployment assumptions change.

