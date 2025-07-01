# Credit Scoring Business Understanding

## 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord reinforces the importance of quantifying risk accurately and holding sufficient capital to cover potential credit losses. This impacts model development in several critical ways:

- **Regulatory Transparency:** Basel II requires banks to demonstrate how risks are measured and managed. Credit scoring models must be interpretable and traceable, especially when estimating key metrics like Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD).

- **Internal Risk Management:** For a financial institution like Bati Bank, a clear model helps internal teams understand which features drive risk. This informs loan approval policies, capital allocation, and customer segmentation.

- **Auditability & Governance:** Basel II promotes strong governance frameworks. Documented and explainable models simplify audits, ensure model validation, and support stress testing.

In the context of our buy-now-pay-later initiative, explainability is essential not just for compliance, but also to maintain trust with regulators, partners, and customers.

## 2. Why create a proxy variable for default in absence of a direct label, and what are the business risks of using such a proxy?

The Xente dataset lacks a clear "default" label, yet supervised learning requires a target variable. To proceed, we construct a proxy for default risk using behavioral signals like RFM (Recency, Frequency, Monetary) and transactional summaries (S).

### Why it's necessary:
- **Model Training Enablement:** A proxy label allows us to train classification models that mimic default risk using available data.
- **Behavior-Driven Insights:** This aligns with the project goal—to convert behavioral signals into actionable credit risk predictions.

### Business risks include:
- **Inaccuracy:** The proxy might not fully reflect real-world defaults, potentially leading to false positives or negatives.
- **Bias Introduction:** Poorly designed proxies could embed systemic bias against specific demographics or behaviors.
- **Regulatory Pushback:** Basel-compliant systems demand clarity in how risk is defined. An opaque or weak proxy could lead to compliance issues or reputational damage if customers are misclassified.

To mitigate this, the proxy will be validated with domain experts and monitored post-deployment to ensure it aligns with actual outcomes.

## 3. What are the trade-offs between using simple, interpretable models (e.g., Logistic Regression with WoE) and complex models (e.g., Gradient Boosting) in a regulated financial context?

In regulated environments like financial services, the choice of model is a strategic balance between regulatory compliance and predictive performance:

### Simple Models (e.g., Logistic Regression + Weight of Evidence)

**Pros:**

- High interpretability: Easy for regulators and stakeholders to understand.
- WoE encoding provides monotonicity and makes variables regulation-friendly.
- Low computational cost; great for deployment in resource-constrained systems.

**Cons:**

- Might oversimplify relationships, missing non-linear trends or complex interactions.
- Lower predictive accuracy in datasets with hidden patterns.

### Complex Models (e.g., Gradient Boosting Machines)

**Pros:**

- Capture non-linearities and feature interactions with high accuracy.
- Often yield better performance on imbalanced or noisy datasets, common in fintech.

**Cons:**

- Difficult to explain: Less transparency, which can violate Basel II’s emphasis on model clarity.
- Requires careful tuning and validation to avoid overfitting.
- May need post-hoc explainability tools (e.g., SHAP, LIME) to meet regulatory demands.

Ultimately, the choice depends on the use case and risk appetite. In this project, we begin with simple, interpretable models for baseline trust and compliance, then compare them with complex models for performance gains, using explainability tools to bridge the gap.
