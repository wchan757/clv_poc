# **Resistance Score (RS_total) & Transaction Utility Score (TUS) Calculation – G-Fiber Cross CLS**

## **📌 Overview**
This document provides a **detailed explanation of the mathematical framework** used to calculate **Resistance Score (RS_total)** and **Transaction Utility Score (TUS)** for customers in the **G-Fiber Cross CLS** system. These metrics are designed to assess **customer engagement, resistance to churn, and financial viability**.  

Additionally, we introduce **Customer Segmentation**, a **two-step approach** where customers are **first categorized based on value**, followed by further **clustering using Gaussian Mixture Models (GMM)** to refine customer groups for targeted marketing.

---

## **1️⃣ Resistance Score (RS_total) Calculation**
### **1.1 What is the Resistance Score?**
The **Resistance Score (RS_total)** quantifies the level of **friction or difficulty a customer faces** when engaging with the service. It incorporates two major categories:

1. **Internal Resistance Factors (internal_z)**  
   - Derived from the **customer’s personality traits**, which influence behavioral tendencies.

2. **External Resistance Factors (external_z)**  
   - Factors **beyond personality**, such as **tenure, payment model, service history, geographic influences, and engagement behavior**.

3. **Ensuring RS_total is Always Positive**  
   - To avoid division errors and improve interpretability, the score is **adjusted to be strictly positive**.

A **higher RS_total** represents a **higher level of resistance**, while a **lower RS_total** indicates **smoother customer engagement**.

---

### **1.2 Step-by-Step Calculation of RS_total**
#### **Step 1: Compute Internal Resistance Score (internal_raw)**
This step quantifies a customer’s **psychological resistance** based on their **Big Five Personality Traits**:

| Personality Trait | Impact on Resistance | Transformation Formula |
|------------------|---------------------|------------------------|
| **Openness (O)** | Higher → **reduces resistance** | \( 6 - O \) |
| **Agreeableness (A)** | Higher → **reduces resistance** | \( 6 - A \) |
| **Conscientiousness (C)** | Higher → **increases resistance** | \( +C \) |
| **Extraversion (E)** | Higher → **increases resistance** | \( +E \) |
| **Neuroticism (N)** | Higher → **increases resistance** | \( +N \) |

\[
\text{internal\_raw} = (6 - O\_pred) + (6 - A\_pred) + C\_pred + E\_pred + N\_pred
\]

✅ **Interpretation:**
- **Higher internal_raw** → Customer is **more resistant** to engagement.
- **Lower internal_raw** → Customer is **more adaptable and open**.

---

#### **Step 2: Compute External Resistance Factors**
External resistance is derived from **customer tenure, payment type, and other contextual elements**. These factors are standardized similarly to **internal resistance**.

---

#### **Step 3: Compute Final Resistance Score (RS_total)**
The total **resistance score** combines internal and external factors:

\[
\text{RS\_total} = \text{RS\_z} + (\lvert \min(\text{RS\_z}) \rvert + 0.001)
\]

✅ **Why Adjust RS_total?**
- Ensures **no negative values**, preventing division errors in later calculations.
- Provides **a stable scoring mechanism** for further analysis.

---

## **2️⃣ Transaction Utility Score (TUS) Calculation**
### **2.1 What is TUS?**
The **Transaction Utility Score (TUS)** measures **customer profitability relative to their resistance**:

\[
TUS = \frac{\text{CLV\_OVERALL\_REVENUE}}{\text{RS\_total}}
\]

where:
- **CLV_OVERALL_REVENUE** = Lifetime customer revenue.
- **RS_total** = Adjusted Resistance Score.

✅ **Key Properties:**
- **Higher TUS** → Customer is **profitable relative to resistance**.
- **Lower TUS** → Customer has **high resistance relative to revenue**.
- **Negative TUS** → Customer is **financially non-viable**.

---

### **2.2 How to Interpret TUS?**
| TUS Score Range | Interpretation |
|----------------|---------------|
| **TUS ≥ 1.0** | Profitable, low resistance |
| **0 < TUS < 1.0** | Profitable but moderate resistance |
| **-1.0 ≤ TUS ≤ 0** | Unprofitable but low resistance |
| **TUS ≤ -1.0** | High resistance, low profitability (churn risk) |

---

## **3️⃣ Customer Segmentation – A Two-Step Approach**
### **3.1 Why Customer Segmentation?**
Customer segmentation is an **essential last step** in the pipeline, where customers are **first categorized based on value (TUS-based segmentation)**, followed by **further refinement using Gaussian Mixture Models (GMM)** to create more targeted marketing groups.

✔ **Personalized Marketing** – Tailor offers and communication for specific segments.  
✔ **Risk Management** – Identify high-risk customers and take proactive actions.  
✔ **Resource Optimization** – Allocate resources efficiently based on segment behavior.  

---

### **3.2 Two-Step Customer Segmentation Approach**
#### **Step 1: Initial Segmentation Based on Value (TUS-based)**
Customers are first categorized into broad **value-based segments**:

| **TUS Value Range** | **Segment Name** | **Description** |
|-----------------|-----------------|-----------------|
| **TUS ≥ 1.0** | **High-Value Customers** | Profitable, low resistance |
| **0 < TUS < 1.0** | **Medium-Value Customers** | Profitable but moderate resistance |
| **TUS ≤ 0** | **Low-Value Customers** | High resistance, low profitability |

---

#### **Step 2: Refining Segmentation Using Gaussian Mixture Model (GMM)**
Within each **TUS-based segment**, we apply **Gaussian Mixture Models (GMM)** to further cluster customers into **three behavioral groups**:

| Segment | Characteristics |
|---------|----------------|
| **Cluster 1 (Engaged & Loyal)** | Long tenure, low resistance, high CLV |
| **Cluster 2 (Moderate Engagement)** | Medium tenure, balanced resistance |
| **Cluster 3 (High-Risk Customers)** | Short tenure, high resistance, low revenue |

\[
\text{Final Clusters} = \text{GMM}(\text{TUS\_Group\_Features})
\]

✅ **Why Use GMM?**
- Captures **hidden behavioral patterns**.
- Provides **soft clustering**, meaning customers can belong to different clusters with probabilities.
- Enables **better targeting for marketing & retention strategies**.

---

## **4️⃣ Summary of Key Metrics & Segmentation**
| Metric | Formula | Purpose |
|--------|---------|---------|
| **Resistance Score (RS_total)** | \( RS_z + |\min(RS_z)| + 0.001 \) | Adjusted resistance |
| **Transaction Utility Score (TUS)** | \( \frac{\text{CLV}}{\text{RS\_total}} \) | Profitability vs. resistance |
| **Step 1: Value-Based Segmentation** | **TUS-based grouping** | Broad segmentation |
| **Step 2: Behavioral Segmentation** | **GMM-based clustering** | Further refinement |

---

## **🚀 Why This Matters**
📌 **Enhances Retention Strategies** – Identify high-risk customers early.  
📌 **Optimizes Marketing** – Personalize communication for each segment.  
📌 **Increases Revenue Efficiency** – Target profitable customers strategically.  

📢 **Final Thought:**  
By incorporating a **two-step segmentation process**, we **first segment customers by value**, then refine these segments using **GMM-based behavioral clustering**. This approach ensures **precision in customer targeting** and **maximizes business impact.** 🚀
