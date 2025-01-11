# **Resistance Score (RS_total) & Transaction Utility Score (TUS) Calculation – G-Fiber Cross CLS**

## **📌 Overview**
This document provides a **detailed explanation of the mathematical framework** used to calculate **Resistance Score (RS_total)** and **Transaction Utility Score (TUS)** for customers in the **G-Fiber Cross CLS** system. These metrics are designed to assess **customer engagement, resistance to churn, and financial viability**.

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

#### **Step 2: Standardize Internal Score (internal_z)**
Since raw scores are **not directly comparable across different customers**, we apply **z-score standardization**:

\[
\text{internal\_z} = \frac{\text{internal\_raw} - \mu}{\sigma}
\]

where:
- \( \mu \) = mean of **internal_raw**.
- \( \sigma \) = standard deviation.

✅ **Why Standardize?**
- Ensures a **consistent scale** across different customer groups.
- Allows **fair comparisons**.

---

#### **Step 3: Compute External Resistance Factors**
External resistance is derived from **customer tenure, payment type, and other contextual elements**.

##### **A) Tenure Score (tenure_z)**
Tenure reflects **customer loyalty and stability**. Longer tenure **reduces resistance**.

\[
\text{tenure\_z} = \frac{\text{TENURE\_COUNT\_MOS} - \mu}{\sigma}
\]

✅ **Interpretation:**
- **Higher tenure_z** → Longer tenure → **Lower resistance**.
- **Lower tenure_z** → Newer customer → **Higher resistance**.

---

##### **B) Payment Score (payment_z)**
Payment method can indicate customer **commitment levels**:
- **Postpaid customers (code = 2)** tend to have **lower friction**.
- **Prepaid or alternative payment types** may indicate **higher friction**.

\[
\text{payment\_raw} =
\begin{cases}
1.0, & \text{if postpaid (2)} \\
3.0, & \text{otherwise (prepaid)}
\end{cases}
\]

Then, we standardize it:

\[
\text{payment\_z} = \frac{\text{payment\_raw} - \mu}{\sigma}
\]

✅ **Why?**
- **Lower payment_z** → Postpaid (lower resistance).
- **Higher payment_z** → Prepaid (higher resistance).

---

##### **C) Other External Resistance Factors**
Additional external factors may include:
- **Service Issues & Complaints**: More issues → **Higher resistance**.
- **Engagement Level**: Lower engagement → **Higher resistance**.
- **Competition in Region**: More competition → **Higher resistance**.

These variables can be standardized the same way as tenure and payment scores.

---

#### **Step 4: Compute Final Resistance Score (RS_z)**
The total **resistance score** combines internal and external factors:

\[
\text{RS\_z} = \text{internal\_z} + \text{payment\_z} - \text{tenure\_z} + \text{other external factors}
\]

✅ **Breakdown:**
- **Higher internal_z** → Increases resistance.
- **Higher payment_z** → Increases resistance.
- **Higher tenure_z** → Decreases resistance.

---

#### **Step 5: Ensure RS_total is Always Positive**
To prevent division errors when calculating TUS:

\[
\text{RS\_total} = \text{RS\_z} + (\lvert \min(\text{RS\_z}) \rvert + 0.001)
\]

✅ **Why?**
- Avoids **zero or negative values**.
- Ensures **stability in calculations**.

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

## **3️⃣ Why These Metrics Matter?**
- **Identifying Retention Risks:** High **RS_total & Low TUS** → Churn Risk.
- **Optimizing Customer Strategies:** Low **RS_total & High TUS** → VIP Treatment.
- **Personalized Engagement:** Helps **target customers efficiently**.
- **Pricing & Promotion Decisions:** Adjust **marketing budgets based on resistance levels**.

---

## **4️⃣ Summary of Key Formulas**
| Metric | Formula | Purpose |
|--------|---------|---------|
| **Internal Resistance Score** | \( (6 - O) + (6 - A) + C + E + N \) | Psychological resistance |
| **Standardized Internal Score** | \( \frac{\text{internal\_raw} - \mu}{\sigma} \) | Normalized resistance |
| **Tenure Score** | \( \frac{\text{TENURE\_COUNT\_MOS} - \mu}{\sigma} \) | Loyalty effect on resistance |
| **Payment Score** | Standardized payment behavior | Financial friction effect |
| **Final Resistance Score** | \( \text{RS\_total} = RS_z + |\min(RS_z)| + 0.001 \) | Ensures stability |
| **Transaction Utility Score** | \( \frac{\text{CLV}}{\text{RS\_total}} \) | Profitability vs. resistance |

🚀 **These calculations drive effective data-driven decision-making for customer engagement and retention.**
