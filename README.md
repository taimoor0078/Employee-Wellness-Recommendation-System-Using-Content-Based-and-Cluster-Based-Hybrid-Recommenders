# RetailRecX

## An Explainable and Ethical E-Commerce Recommendation System

RetailRecX is a machine-learning-based e-commerce recommendation system designed to provide personalised product recommendations while addressing **explainability, transparency, popularity bias, diversity, and ethical recommendation practices**.

The project combines a **Random Forest recommendation model** with **SHAP** and **LIME** explainability techniques. It also investigates popularity bias and evaluates two mitigation strategies: **popularity-aware re-ranking** and **long-tail re-ranking**.

---

## Project Overview

Traditional recommender systems can behave like black boxes: they may recommend products without clearly explaining why a particular item was selected.

RetailRecX addresses this problem by combining:

- Personalised product recommendation
- Machine learning
- Explainable AI (XAI)
- SHAP explanations
- LIME explanations
- Popularity-bias analysis
- Popularity-aware re-ranking
- Long-tail re-ranking
- Catalogue coverage and diversity evaluation
- Ethical recommendation analysis
- A controlled Generative AI explanation layer

The system uses historical e-commerce transaction data to generate customer-specific product recommendations and provide understandable explanations for those recommendations.

---

## Key Objectives

1. Develop a personalised e-commerce recommendation system.
2. Use machine learning to estimate the likelihood that a customer will purchase a product.
3. Provide explanations for individual recommendations using SHAP and LIME.
4. Identify and analyse popularity bias in recommendations.
5. Evaluate mitigation strategies for reducing excessive popularity concentration.
6. Measure the trade-offs between recommendation accuracy, diversity, catalogue coverage, and bias reduction.
7. Explore how Generative AI can be used as a controlled explanation layer without replacing the underlying recommendation model or XAI evidence.

---

## System Architecture

The RetailRecX pipeline follows these main stages:

```text
                    ┌──────────────────────┐
                    │   Online Retail      │
                    │       Dataset        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Cleaning &      │
                    │ Transformation       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    │ Customer / Product / │
                    │ Interaction Features │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Random Forest      │
                    │ Recommendation Model │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Product Scoring &    │
                    │ Top-K Ranking        │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
       ┌────────────────┐           ┌────────────────┐
       │ SHAP           │           │ LIME           │
       │ Explanation    │           │ Explanation    │
       └────────┬───────┘           └───────┬────────┘
                └──────────────┬────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Bias Analysis &      │
                    │ Mitigation           │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
       Popularity-Aware                  Long-Tail
         Re-ranking                     Re-ranking
                │                             │
                └──────────────┬──────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Personalised Top-K   │
                    │ Recommendations +   │
                    │ Explanations         │
                    └──────────────────────┘
```

The architecture shown in the project report also includes data storage, Python-based processing, machine-learning scoring, SHAP/LIME explainability, API/deployment components, a visualisation dashboard, evaluation metrics, and a monitoring/feedback loop.

---

## Dataset

RetailRecX uses the **Online Retail dataset** containing historical e-commerce transaction records.

### Original Dataset

- **Transactions:** 541,909
- **Attributes:** 8

The original attributes include:

| Attribute | Description |
|---|---|
| Invoice Number | Transaction/invoice identifier |
| Stock Code | Product identifier |
| Product Description | Product description |
| Quantity | Quantity purchased |
| Invoice Date | Date/time of transaction |
| Unit Price | Product unit price |
| Customer ID | Customer identifier |
| Country | Customer country |

### Data Cleaning

The preprocessing stage removes unsuitable records, including:

- Missing customer identifiers
- Cancelled transactions
- Invalid quantities
- Invalid prices
- Other unsuitable transaction records

After preprocessing, the project dataset contained:

- **391,283 transactions**
- **4,334 customers**
- **3,660 products**

Invoice dates were converted into an appropriate date format, and customer-product interaction features were created.

---

## Feature Engineering

RetailRecX uses three main feature groups.

### 1. Customer Features

Customer-level features represent purchasing behaviour, including:

- Total purchases
- Total spending
- Purchase frequency
- Recency
- Other purchasing activity indicators

### 2. Product Features

Product-level features represent product behaviour, including:

- Product popularity
- Customer reach
- Purchase quantity
- Other product-level characteristics

### 3. Customer-Product Interaction Features

Interaction features capture the relationship between an individual customer and a product, including:

- Customer-product recency
- Customer-product spending
- Previous purchase count
- Previous interaction
- Other interaction-based behaviour

These features allow the recommendation model to combine customer behaviour, product characteristics, and previous customer-product interactions.

---

## Recommendation Model

RetailRecX uses a **Random Forest classifier** to estimate the likelihood that a customer will purchase a particular product.

The model:

1. Receives customer, product, and interaction features.
2. Estimates purchase likelihood.
3. Generates recommendation scores.
4. Ranks candidate products according to predicted scores.
5. Produces personalised Top-K recommendations.

### Model Performance

The recommendation model achieved a **ROC-AUC of 0.8862** on the validation data.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.8862 |
| Precision@10 | 0.3251 |
| Recall@10 | 0.2046 |
| NDCG@10 | 0.3850 |
| Hit Rate@10 | 0.8671 |
| Catalogue Coverage | 29.93% |
| Diversity | 0.0824 |

The recommendation evaluation used an evaluation set containing **1,317 customers**.

---

# Explainable AI

RetailRecX uses two complementary XAI methods:

- SHAP
- LIME

The purpose is to identify the factors influencing individual recommendations and make the recommendation process more transparent.

---

## SHAP

**SHAP (SHapley Additive exPlanations)** is used to explain how individual features contribute to the recommendation score generated by the Random Forest model.

A positive contribution indicates that a feature increases the recommendation likelihood, while a negative contribution indicates that it decreases the likelihood.

### Important Features Identified by SHAP

| Rank | Feature | Interpretation |
|---:|---|---|
| 1 | Customer-Product Recency | Recent interaction increases recommendation relevance |
| 2 | Customer-Product Spending | Higher historical spending indicates stronger preference |
| 3 | Previous Purchase Count | Repeated purchases indicate customer interest |
| 4 | Previous Interaction | Existing interaction supports recommendation |
| 5 | Product Customer Reach | Products purchased by more customers receive stronger influence |
| 6 | Total Quantity Purchased | Higher purchase quantity indicates product preference |

A local SHAP explanation can also be generated for an individual customer-product recommendation.

---

## LIME

**LIME (Local Interpretable Model-agnostic Explanations)** is used to explain individual recommendations.

Instead of explaining the complete model, LIME approximates model behaviour around a selected customer-product instance and identifies the feature conditions that most strongly influence the prediction.

This provides a human-interpretable local explanation for an individual recommendation.

---

## SHAP vs LIME

SHAP and LIME provide complementary explanations.

| Aspect | SHAP | LIME |
|---|---|---|
| Explanation Type | Global and local | Mainly local |
| Approach | Feature contribution values | Local surrogate model |
| Interpretation | Strong theoretical foundation | Human-interpretable local approximation |
| Recommendation Explanation | Feature contribution to score | Influential conditions/features |
| Agreement | — | 5 of top 6 features overlap |

For the selected recommendation, **5 of the top 6 important features overlapped between SHAP and LIME**, indicating substantial agreement between the two explanation techniques.

---

# Ethical Analysis

## Popularity Bias

A major ethical issue analysed in RetailRecX is **popularity bias**.

Popularity bias occurs when a recommendation system gives disproportionately high exposure to products that are already popular while less frequently purchased products receive fewer recommendation opportunities.

In the baseline system:

- The top 20% of products represented only **20% of the catalogue**.
- However, these products occupied **88.64% of recommendation slots**.
- Baseline catalogue coverage was **29.93%**.
- Baseline diversity was **0.0824**.

This demonstrates a strong concentration of recommendations around highly popular products.

---

# Bias Mitigation Strategies

RetailRecX evaluates two re-ranking approaches.

## 1. Popularity-Aware Re-ranking

A popularity penalty is introduced during ranking to reduce the influence of highly popular products.

### Results

- Top-20% popularity share: **88.64% → 25.15%**
- Catalogue coverage: **29.93% → 48.80%**
- Diversity: **0.0824 → 0.1343**
- Precision@10: **0.3251 → 0.2128**
- NDCG@10: **0.3850 → 0.2510**

This approach substantially reduces popularity concentration while increasing catalogue coverage and diversity, but it also reduces recommendation accuracy.

---

## 2. Long-Tail Re-ranking

The long-tail strategy reserves a proportion of recommendation positions for less popular products.

### Results

- Top-20% popularity share: **88.64% → 59.13%**
- Catalogue coverage: **29.93% → 35.83%**
- Diversity: **0.0824 → 0.0986**
- Precision@10: **0.3251 → 0.3000**
- NDCG@10: **0.3850 → 0.3154**
- Hit Rate@10: **0.8671 → 0.8633**

### Overall Comparison

| Metric | Baseline | Popularity-Aware Re-ranking | Long-Tail Re-ranking |
|---|---:|---:|---:|
| Top-20% Popularity Share | 88.64% | 25.15% | 59.13% |
| Precision@10 | 0.3251 | 0.2128 | 0.3000 |
| NDCG@10 | 0.3850 | 0.2510 | 0.3154 |
| Hit Rate@10 | 0.8671 | — | 0.8633 |
| Catalogue Coverage | 29.93% | 48.80% | 35.83% |
| Diversity | 0.0824 | 0.1343 | 0.0986 |

The experiments demonstrate that ethical recommendation design involves trade-offs between relevance, diversity, catalogue coverage, and popularity concentration.

---

# Generative AI and Ethical Recommendation

RetailRecX considers Generative AI as an **additional explanation layer**, rather than as the component responsible for selecting products.

The proposed approach is:

```text
Customer Data
     │
     ▼
Recommendation Model
     │
     ▼
Product Recommendation
     │
     ├──────────────► SHAP Evidence
     │
     └──────────────► LIME Evidence
                         │
                         ▼
                Verified Explanation
                         │
                         ▼
                 Generative AI Layer
                         │
                         ▼
              Natural-Language Explanation
```

The Generative AI layer could convert technical SHAP/LIME evidence into accessible natural-language explanations.

For example, it could explain why a product matches a customer's previous purchasing behaviour or allow users to ask questions about a recommendation.

### Ethical Risks

Introducing Generative AI also creates additional risks:

- Hallucinated explanations
- Unsupported recommendation reasoning
- Reproduction of existing recommendation biases
- Privacy concerns
- Exposure of sensitive purchasing information
- Overly persuasive explanations
- Misleading natural-language explanations

Therefore, the proposed architecture keeps the machine-learning recommender and XAI methods as the source of evidence. Generative AI should only transform verified evidence into natural-language explanations.

---

# Evaluation Framework

RetailRecX evaluates the recommendation system using multiple dimensions rather than predictive performance alone.

### Predictive Performance

- ROC-AUC
- Precision@10
- Recall@10
- NDCG@10
- Hit Rate@10

### Recommendation Quality

- Catalogue Coverage
- Diversity

### Explainability

- SHAP feature contributions
- LIME local explanations
- SHAP/LIME feature agreement

### Ethical Evaluation

- Top-20% popularity share
- Catalogue coverage changes
- Diversity changes
- Accuracy/diversity trade-offs
- Popularity concentration

---

# Technology / System Components

The architecture presented in the project includes the following conceptual components:

- Online Retail dataset
- Data cleaning and preprocessing
- Python-based processing
- Feature engineering
- Random Forest model
- SHAP
- LIME
- Recommendation scoring and ranking
- Popularity-bias analysis
- Re-ranking strategies
- Evaluation metrics
- API/deployment layer
- Visualisation/dashboard
- Data storage
- Monitoring and feedback loop

---

# Project Workflow

```text
1. Load Online Retail Dataset
             │
             ▼
2. Clean Transaction Data
             │
             ▼
3. Transform Dates and Transactions
             │
             ▼
4. Create Customer Features
             │
             ▼
5. Create Product Features
             │
             ▼
6. Create Customer-Product Features
             │
             ▼
7. Train Random Forest Classifier
             │
             ▼
8. Generate Product Scores
             │
             ▼
9. Generate Top-K Recommendations
             │
             ├───────────────┐
             ▼               ▼
           SHAP             LIME
             │               │
             └───────┬───────┘
                     ▼
             Explain Recommendations
                     │
                     ▼
             Analyse Popularity Bias
                     │
             ┌───────┴────────┐
             ▼                ▼
     Popularity-Aware    Long-Tail
       Re-ranking       Re-ranking
             │                │
             └───────┬────────┘
                     ▼
          Compare Recommendation
               and Ethical Metrics
```

---

# Results Summary

RetailRecX achieved a **ROC-AUC of 0.8862** and **Hit Rate@10 of 0.8671** in the reported evaluation.

The XAI analysis showed that customer-product recency, customer-product spending, previous purchase count, previous interaction, product customer reach, and total quantity purchased were important recommendation factors.

The ethical analysis identified substantial popularity concentration in the baseline system, where products from the top 20% of the catalogue occupied **88.64% of recommendation slots**.

The mitigation experiments showed that re-ranking can reduce popularity concentration and increase catalogue coverage and diversity, although these changes involve trade-offs with recommendation accuracy.

---

# Limitations

The project identifies several limitations:

1. The Online Retail dataset contains historical transaction information and does not include demographic or detailed preference information.
2. The system may face cold-start problems for new customers and products with little or no historical interaction.
3. Customer-product interactions are relatively sparse.
4. Evaluation is based on historical offline transaction data rather than real-time user interactions.
5. Offline metrics may not fully represent how users would respond to recommendations in a real e-commerce environment.
6. Ethical findings depend on the characteristics of the selected dataset and may differ for other customer populations or product catalogues.
7. SHAP and LIME improve interpretability but do not guarantee that users will understand explanations correctly.
8. A Generative AI explanation layer could introduce hallucinations or misleading explanations if it is not grounded in verified model evidence.

---

# Conclusion

RetailRecX demonstrates a practical approach to building an **explainable and ethically aware personalised e-commerce recommendation system**.

The project combines machine-learning-based recommendation with SHAP and LIME explanations to improve transparency. It also evaluates popularity bias and demonstrates how re-ranking strategies can change the balance between recommendation relevance, catalogue coverage, diversity, and popularity concentration.

A key design principle of the project is that **Generative AI should not replace the underlying recommendation model or XAI evidence**. Instead, it can operate as a controlled explanation layer that converts verified recommendation evidence into understandable natural-language explanations.

Overall, RetailRecX evaluates recommender systems from multiple perspectives, including:

- Personalisation
- Predictive performance
- Explainability
- Diversity
- Catalogue coverage
- Popularity bias
- Fair exposure
- Transparency
- User trust
- Ethical use of Generative AI

---

## Project Information

**Project:** RetailRecX  
**Description:** An Explainable and Ethical E-Commerce Recommendation System  
**Student:** Taimoor Ahmad  
**Programme:** MSc Data Science  
**Institution:** Dublin Business School  
**Academic Assessment:** CA2  
**Date:** August 2025

---

## Academic Note

This repository documents the RetailRecX project and its reported methodology, architecture, evaluation results, explainability analysis, ethical analysis, and Generative AI discussion.

