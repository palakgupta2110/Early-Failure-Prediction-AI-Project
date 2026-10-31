# DATA SCIENCE MINOR PROJECT REPORT

---

<br><br><br>

<div align="center">

### DATA SCIENCE

## **PROJECT REPORT**

### (Project Semester January-April 2025)

<br><br>

# **EARLY FAILURE PREDICTION AI: PREDICTING MACHINE FAILURES BEFORE THEY HAPPEN USING AI**

<br><br>

**Submitted by**

**[Student Name]**  
**Registration No.:** [Registration Number]  
**Programme and Section:** B.Tech Computer Science and Engineering (CSE), [Section]  
**Course Code:** [Course Code]  

<br><br>

**Under the Guidance of**  
**[Name of Faculty Coordinator]**  
UID: [Faculty UID]  
Designation: [Assistant Professor / Associate Professor]  

<br><br>

**Discipline of CSE/IT**  
**Lovely School of Computer Science and Engineering**  
**Lovely Professional University, Phagwara**

</div>

<br><br><br><br>

---

## CERTIFICATE

This is to certify that **[Student Name]** bearing Registration no. **[Registration Number]** has completed **[Course Code]** project titled, **“EARLY FAILURE PREDICTION AI: PREDICTING MACHINE FAILURES BEFORE THEY HAPPEN USING AI”** under my guidance and supervision. To the best of my knowledge, the present work is the result of his/her original development, effort and study.

<br><br><br>

**Signature and Name of the Supervisor:** ___________________________________  
**Designation of the Supervisor:** [Designation of Supervisor]  
**School of Computer Science and Engineering**  
**Lovely Professional University**  
**Phagwara, Punjab.**  

<br>

**Date:** [Submission Date]

---

## DECLARATION

I, **[Student Name]**, student of **Bachelor of Technology in Computer Science and Engineering (B.Tech CSE)** under CSE/IT Discipline at, Lovely Professional University, Punjab, hereby declare that all the information furnished in this project report is based on my own intensive work and is genuine.

<br><br><br>

**Date:** [Submission Date]  
**Registration No.:** [Registration Number]  

<br>

**Signature:** ___________________________________  
**Name of the Student:** [Student Name]  

---

## ACKNOWLEDGEMENT

I would like to express my sincere gratitude and indebtedness to my esteemed project supervisor, **[Name of Faculty Coordinator]**, for continuous guidance, constructive criticism, and invaluable technical suggestions throughout the formulation and implementation of this minor project. Their insightful feedback on industrial predictive maintenance, sensor feature aggregation, and machine learning model validation was critical to the success of this research.

I extend my heartfelt appreciation to the **Discipline of Computer Science and Engineering**, **Lovely School of Computer Science and Engineering**, and **Lovely Professional University, Phagwara** for providing world-class computational facilities, digital libraries, and an enriching academic environment.

Finally, I express my profound thanks to my family, peers, and mentors whose constant encouragement and cooperation supported me at every phase of this work.

<br>

**[Student Name]**  
Department of Computer Science and Engineering  
Lovely Professional University

---

## TABLE OF CONTENTS

- **Cover Page**
- **Certificate**
- **Declaration**
- **Acknowledgement**
- **List of Figures**
- **List of Tables**

### 1. Introduction
- 1.1 Background of Predictive Maintenance (PdM) & Industrial IoT
- 1.2 Motivation: Compressor Systems in Urban Metro Transit
- 1.3 Problem Statement
- 1.4 Objectives of the Study
- 1.5 Scope and Delimitations of Research
- 1.6 Report Organization

### 2. Source of Dataset
- 2.1 Dataset Provenance & UCI Repository Attribution
- 2.2 Physical System Architecture: Metro Air Generation Unit (AGU)
- 2.3 Sensor Telemetry Inventory (16 Physical Channels)
- 2.4 Data Acquisition Protocol & Temporal Horizon
- 2.5 Data Quality, Sparsity, and Operating Regimes

### 3. Exploratory Data Analysis (EDA) & Preprocessing
- 3.1 Raw Ingestion & Integrity Auditing (1.51M Records)
- 3.2 Missing Value Profiling & Imputation Strategy
- 3.3 Temporal Resampling & 5-Minute Window Feature Extraction
- 3.4 Feature Engineering (Mean, Std Dev, Min, Max $\rightarrow$ 64 Dimensions)
- 3.5 Multivariate Proxy Anomaly Formulation (Median Absolute Deviation)
- 3.6 Severe Class Imbalance (98% Normal vs. 2% Anomaly Proxy)
- 3.7 Train / Validation / Test Stratified Partitioning Protocol

### 4. Analysis on Dataset (Classifier Comparative Benchmarks)
- **4.1 Analysis 1: K-Nearest Neighbors (KNN)**
  - 4.1.i Introduction
  - 4.1.ii General Description & Mathematical Formulation
  - 4.1.iii Specific Requirements, Hyperparameters & Distance Metrics
  - 4.1.iv Analysis Results & Confusion Matrix
  - 4.1.v Visualization & Sensitivity Dynamics
- **4.2 Analysis 2: Support Vector Machine (SVM)**
  - 4.2.i Introduction
  - 4.2.ii General Description & RBF Kernel Geometry
  - 4.2.iii Specific Requirements, Dual Optimization & Class Weighting
  - 4.2.iv Analysis Results & Margin Metrics
  - 4.2.v Visualization & Support Vector Behavior
- **4.3 Analysis 3: Gaussian Naive Bayes (GNB)**
  - 4.3.i Introduction
  - 4.3.ii General Description & Conditional Independence Principle
  - 4.3.iii Specific Requirements, Gaussian Likelihood & Prior Balancing
  - 4.3.iv Analysis Results & False Alarm Manifestation
  - 4.3.v Visualization & Likelihood Distributions
- **4.4 Analysis 4: Logistic Regression (Optimal Classifier)**
  - 4.4.i Introduction
  - 4.4.ii General Description & Log-Odds Optimization
  - 4.4.iii Specific Requirements, Regularization & Threshold Tuning
  - 4.4.iv Analysis Results & Confusion Matrix
  - 4.4.v Visualization & Score vs. Threshold Gauge
- **4.5 Analysis 5: Decision Tree Classifier**
  - 4.5.i Introduction
  - 4.5.ii General Description & Recursive Binary Splitting
  - 4.5.iii Specific Requirements, Impurity Criteria & Pruning
  - 4.5.iv Analysis Results & Confusion Matrix
  - 4.5.v Visualization & Tree Partition Dynamics
- **4.6 Cross-Model Comparative Synthesis & Evaluation Discussion**
  - 4.6.1 Leaderboard Ranking (F1, PR-AUC, Recall, Precision, Balanced Accuracy)
  - 4.6.2 Why Accuracy Fails: The 98% Base-Rate Fallacy
  - 4.6.3 Precision-Recall Tradeoff Frontier
  - 4.6.4 Algorithmic Rationale: Why Logistic Regression Won

### 5. Conclusion
- 5.1 Project Summary
- 5.2 Key Technical Takeaways
- 5.3 Methodological Integrity Regarding Proxy Labels

### 6. Future Scope
- 6.1 Integration of Certified Run-to-Failure Physical Logs
- 6.2 Deep Temporal Architectures (LSTM, GRU, TCN)
- 6.3 Edge SCADA Integration & Industrial Stream Processing
- 6.4 Explainable AI (SHAP / Integrated Gradients) for Maintenance Crews

### 7. References

---

## LIST OF FIGURES

- **Figure 1.1**: Conceptual Architecture of Industrial Predictive Maintenance Pipeline
- **Figure 2.1**: Schematic of MetroPT-3 Air Production & Treatment Unit (APU/AGU)
- **Figure 3.1**: Raw Sensor Telemetry Trends across 8-Month Operational Horizon
- **Figure 3.2**: Correlation Matrix Heatmap across 16 Primary Continuous Channels
- **Figure 3.3**: Multivariate Proxy Anomaly Distribution (98th Percentile Cutoff)
- **Figure 4.1**: KNN Confusion Matrix Heatmap on Held-Out Test Set
- **Figure 4.2**: SVM Non-Linear Margin Classification Boundary & Confusion Matrix
- **Figure 4.3**: Gaussian Naive Bayes Prediction Distribution Showing False Alarms
- **Figure 4.4**: Logistic Regression Threshold Gauge & Diagnostic Confusion Matrix
- **Figure 4.5**: Decision Tree Feature Importance Hierarchy
- **Figure 4.6**: Cross-Classifier Test Metric Comparison Bar Chart (F1, Recall, Precision, PR-AUC)
- **Figure 4.7**: Precision vs. Recall Frontier across the Five Machine Learning Classifiers

---

## LIST OF TABLES

- **Table 2.1**: MetroPT-3 Telemetry Channel Inventory (Sensor Descriptions, Units, Physical Categories)
- **Table 3.1**: Descriptive Statistics of Key Sensor Channels (Mean, Std, Min, 25%, Median, 75%, Max)
- **Table 3.2**: Feature Engineering Matrix (Summary Statistics per 5-Minute Temporal Window)
- **Table 4.1**: K-Nearest Neighbors Evaluation Metrics on Test Set
- **Table 4.2**: Support Vector Machine Evaluation Metrics on Test Set
- **Table 4.3**: Gaussian Naive Bayes Evaluation Metrics on Test Set
- **Table 4.4**: Logistic Regression Evaluation Metrics on Test Set
- **Table 4.5**: Decision Tree Evaluation Metrics on Test Set
- **Table 4.6**: Comprehensive Model Comparison Leaderboard (All 5 Classifiers Benchmark)
- **Table 4.7**: Confusion Matrix Summary Across All Five Classifiers

---

# 1. INTRODUCTION

### 1.1 Background of Predictive Maintenance (PdM) & Industrial IoT
Modern industrial systems are increasingly dependent on cyber-physical integration, Industrial Internet of Things (IIoT) sensor networks, and edge-to-cloud analytics. Traditional industrial maintenance philosophies fall into two distinct paradigms:
1. **Corrective (Reactive) Maintenance**: Assets are operated continuously until functional failure occurs, followed by emergency repair or replacement. While simple, reactive maintenance incurs catastrophic indirect costs: unscheduled plant shutdowns, severe collateral damage to adjacent machinery, safety hazards for operational personnel, and immense supply-chain interruptions.
2. **Preventive (Time-Based / Run-Hour) Maintenance**: Maintenance actions, overhauls, and component replacements are conducted according to rigid pre-determined calendars or operational hour intervals (e.g., every 5,000 operating hours). Although superior to reactive maintenance, preventive protocols introduce significant economic inefficiencies: components with substantial remaining useful life (RUL) are discarded prematurely, while random component infant-mortality failures remain undetected.

**Predictive Maintenance (PdM)** represents the Industry 4.0 paradigm shift. By continuously collecting, aggregating, and analyzing multi-channel telemetry streams (temperatures, pressures, electrical currents, vibration signals, and valve duty cycles), machine learning algorithms learn baseline nominal operational envelopes. Consequently, incipient degradation, mechanical friction, pneumatic leakage, and thermal stress are identified weeks or months before catastrophic failure manifests.

### 1.2 Motivation: Compressor Systems in Urban Metro Transit
In heavy urban rail systems, such as metropolitan underground train networks, pneumatic compressors serve as critical mission-supporting auxiliary machinery. Compressed air delivered by the **Air Production and Treatment Unit (APU / AGU)** powers three fundamental locomotive safety subsystems:
1. **Frictional and Emergency Braking**: Rail pneumatic braking systems rely entirely on steady reservoir pressure (typically maintained between 8.5 bar and 10.0 bar) to actuate disc brake calipers and emergency failsafe release mechanisms.
2. **Air Suspension Systems**: Secondary air springs stabilize railcar passenger comfort, dampen structural track harmonics, and compensate for dynamic passenger load fluctuations.
3. **Automated Passenger Doors**: Pneumatically actuated doors require instant, uninterrupted pressurized air delivery to cycle reliably at station stops.

A single mid-transit compressor breakdown or sudden pressure decay forces immediate automatic train stopping, network-wide line delays, passenger stranding, and extreme economic penalties for metro operators. Building robust, automated predictive intelligence to detect abnormal compressor operational states from telemetry data is therefore an imperative engineering priority.

### 1.3 Problem Statement
Air compressors deployed on revenue-service metro railcars operate under harsh, dynamic environments: ambient temperature shifts, variable passenger load weights, continuous start-stop duty cycles, and track vibration. Industrial maintenance teams face the challenge of extracting actionable failure warnings from high-frequency, noisy sensor telemetry streams.

The core challenge involves:
- Processing massive telemetry logs containing over **1.5 million** raw multi-sensor readings.
- Overcoming severe class imbalance, where normal operation constitutes over **98%** of the timeline while anomalous degradation windows represent less than **2%**.
- Benchmarking multiple classical machine learning classifiers (KNN, SVM, Gaussian Naive Bayes, Logistic Regression, Decision Tree) to identify the optimal model.
- Navigating the critical limitation of public telemetry datasets: namely, the absence of verified component teardown failure logs, necessitating the mathematically grounded formulation of **multivariate proxy anomaly labels**.

### 1.4 Objectives of the Study
The overarching objectives of this project are:
1. **Telemetry Pipeline Engineering**: Ingest the raw UCI MetroPT-3 compressor dataset (1,516,948 timestamps, 208 MB) and implement high-throughput deterministic temporal aggregation (5-minute windows) reducing data volume to 50,782 stable feature instances.
2. **Feature Extraction**: Extract 64 comprehensive statistical features (Mean, Standard Deviation, Minimum, Maximum across 16 physical sensor channels) capturing operational levels, signal stability, and dynamic peaks.
3. **Robust Proxy Label Formulation**: Engineer a transparent, mathematically defensible multivariate anomaly proxy label based on Median Absolute Deviation (MAD) across all sensor channels, isolating the 98th percentile extreme deviation regime.
4. **Benchmarking Five Classifiers**: Implement, optimize, and evaluate five diverse classification algorithms:
   - K-Nearest Neighbors (Instance-Based Distance Learner)
   - Support Vector Machine (Maximum-Margin Kernel Learner)
   - Gaussian Naive Bayes (Probabilistic Feature-Independent Learner)
   - Logistic Regression (Linear Log-Odds Estimator)
   - Decision Tree (Non-Parametric Recursive Partitioner)
   *(Note: Random Forest is intentionally excluded per academic project specifications).*
5. **Threshold Calibration & Validation**: Formulate an isolated validation threshold tuning mechanism maximizing the $F_1$-score on held-out data without data leakage.
6. **Production-Ready Dashboard Deployment**: Develop a modern, industrial-grade dark analytics dashboard using Streamlit and Plotly, enabling responsive time-series visualization and real-time historical window inference.

### 1.5 Scope and Delimitations of Research
- **Scope**: The project evaluates sensor-level telemetry from urban metro compressors, covering physical thermal, pneumatic, electrical, and control channels. The evaluation utilizes rigorous class-imbalance metrics: $F_1$-score, PR-AUC, Precision, Recall, Balanced Accuracy, and Confusion Matrices.
- **Delimitations**: The dataset reflects revenue telemetry without invasive teardown ground truth; thus, the binary labels represent statistical operational anomalies (proxy indicators), not guaranteed physical mechanical breakdowns. Real-world deployment would require integration with physical maintenance logs and calibrated probabilistic forecast horizons.

### 1.6 Report Organization
The remainder of this report is organized as follows:
- **Section 2** details the provenance, architecture, and physical sensors of the UCI MetroPT-3 dataset.
- **Section 3** presents the Exploratory Data Analysis (EDA), temporal windowing, feature extraction, and proxy-label formulation.
- **Section 4** provides an exhaustive analysis of each of the five classifiers, detailing their algorithmic theory, hyperparameters, confusion matrices, and comparative synthesis.
- **Section 5** synthesizes conclusions and core findings.
- **Section 6** outlines the future engineering scope and extensions.
- **Section 7** lists formal academic and technical references.

---

# 2. SOURCE OF DATASET

### 2.1 Dataset Provenance & UCI Repository Attribution
The dataset utilized in this investigation is the **MetroPT-3 (Air Compressor) Dataset**, hosted publicly on the prestigious **UCI Machine Learning Repository**:
- **Dataset Title**: MetroPT-3 Dataset
- **Repository URI**: `https://archive.ics.uci.edu/dataset/791/metropt3%2Bdataset?utm_source`
- **Subject Category**: Industrial Predictive Maintenance / Cyber-Physical Systems
- **Originating Institution**: Faculty of Engineering, University of Porto (FEUP) & Metro do Porto, Portugal
- **Primary Citation**: Veloso, B. M., et al., "A Real-World Dataset for Multi-Stage Predictive Maintenance in Metro Trains," *Sensors*, 2022.

The data was acquired from a passenger transit train operating on Line Yellow of the Metro do Porto light rail transit network in Porto, Portugal.

### 2.2 Physical System Architecture: Metro Air Generation Unit (AGU)
The monitored equipment is an onboard **Air Production and Treatment Unit (APU / AGU)** situated beneath the railcar chassis. The AGU consists of:
1. **Rotary Screw Compressor**: Electrically driven compressor unit that inducts filtered atmospheric air and compresses it to working pressure.
2. **Electric Drive Motor**: High-power AC induction motor coupled directly to the compressor shaft.
3. **Oil Cooling & Lubrication Circuit**: Circulates synthetic lubricant to seal screw lobes, cool compressed air, and lubricate mechanical bearings.
4. **Air Drying Towers**: Desiccant-based twin-tower adsorption system that strips humidity from compressed air to prevent freezing and corrosion in locomotive brake cylinders.
5. **Drain Valve (DV) & Safety Relief Valves**: Electropneumatic discharge valves that purge condensed moisture and vent excess pressure.
6. **Main Air Reservoirs**: Storage pressure vessels delivering conditioned pressurized air to locomotive braking and suspension manifolds.

### 2.3 Sensor Telemetry Inventory (16 Physical Channels)
The telemetry system logs 16 distinct channels continuously. These encompass analog thermodynamic readings, pneumatic pressures, motor electrical demand, and digital supervisory states:

| Column Name | Physical Phenomenon Measured | Engineering Unit | Sensor Type | Operational Relevance |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | UTC Date & Time | YYYY-MM-DD HH:MM:SS | Telemetry Clock | Temporal ordering & window index |
| `TP2` | Pressure in Compressor Stage 2 | bar | Analog Transducer | Monitors internal compression ratio |
| `TP3` | Compressor Delivery Pressure | bar | Analog Transducer | Output pressure entering drying tower |
| `H1` | High-Pressure Discharge Temp | °C | Thermocouple | Thermal load; overheating indicator |
| `DV_pressure` | Drain Valve Pressure Drop | bar | Analog Transducer | Condensate discharge monitoring |
| `Reservoirs` | Main Reservoir Pressure | bar | Analog Transducer | Train brake pipeline supply pressure |
| `Oil_temperature`| Compressor Lubricant Temp | °C | PT100 Resistance | Mechanical friction & cooler efficiency |
| `Motor_current` | Motor Electrical Current | Ampere (A) | Hall-Effect Current Sensor | Motor load, torque & electrical power |
| `COMP` | Compressor Run Command | Digital (0/1) | PLC Output Relay | High when compressor is pumping |
| `DV_eletric` | Electric Drain Valve State | Digital (0/1) | Solenoid Actuator Relay | Purge valve activation state |
| `Towers` | Dryer Tower Switching | Digital (0/1) | Relay Feedback | Indicates desiccant tower regeneration |
| `MPG` | Motor Protection Group | Digital / Signal | Contact Sensor | Auxiliary electrical safety interlock |
| `LPS` | Low Pressure Switch | Digital (0/1) | Pressure Switch Relay | Closes when pressure drops below safety |
| `Pressure_switch`| Main Control Pressure Switch| Digital (0/1) | Mechanical Switch | Cut-in / cut-out compressor regulator |
| `Oil_level` | Sump Oil Level Switch | Digital (0/1) | Float Switch | Critical low-oil level indicator |
| `Caudal_impulses`| Air Flow Meter Impulses | Integer Count | Impeller Pulse Counter | Volumetric air delivery flow rate |

### 2.4 Data Acquisition Protocol & Temporal Horizon
- **Sampling Frequency**: Sensor readings were digitized and recorded once every **10 seconds** ($0.1\text{ Hz}$).
- **Temporal Horizon**: Telemetry acquisition spanned from **February 1, 2020 00:00:00 UTC** through **September 11, 2020 23:59:50 UTC** (approximately 8 months of continuous operational monitoring).
- **Data Volume**: The uncompressed raw CSV file consumes **208.19 MB** of disk storage and comprises **1,516,948 raw observation records**.

### 2.5 Data Quality, Sparsity, and Operating Regimes
Because the data stems from revenue transit operations, it contains real-world operational complexities:
- Periodic overnight stabling where compressors remain idle and ambient temperatures equilibrate.
- Frequent dynamic cycling between loaded (compressor pumping air into reservoirs) and unloaded (compressor idling or stopped) states.
- Thermal inertia: oil temperature rises gradually under continuous heavy pumping cycles and cools slowly during station dwell times.

---

# 3. EXPLORATORY DATA ANALYSIS (EDA) & PREPROCESSING

### 3.1 Raw Ingestion & Integrity Auditing
Auditing of the raw CSV file (`MetroPT3(AirCompressor).csv`) revealed an initial row count of **1,516,948** with 17 total columns (one row index, one timestamp column, and 15 sensor channels).
- Fast delimiter detection confirmed standard comma-separated format.
- Parsing timestamps with standard ISO-8601 formatting yielded valid temporal indexing spanning 224 calendar days.
- Zero corrupted lines or truncated byte streams were encountered.

### 3.2 Missing Value Profiling & Imputation Strategy
In the raw data stream, transient sensor dropouts or electrical communication packet drops produced a small volume of null and infinite values ($<0.05\%$).
- Infinite values ($+\infty, -\infty$) resulting from sensor open-circuit signals were converted to standard `NaN`.
- To avoid lookahead bias and maintain deterministic pipeline inference, a scikit-learn `SimpleImputer` utilizing the **median strategy** was embedded directly into all model pipelines. The median is robust against extreme sensor spikes compared to the arithmetic mean.

### 3.3 Temporal Resampling & 5-Minute Window Feature Extraction
Training machine learning models directly on 1.5 million 10-second data points is computationally inefficient, noise-prone, and unreflective of maintenance planning intervals. Transient sensor spikes lasting 10 seconds do not signify equipment degradation.

To resolve this, the continuous telemetry was aggregated into non-overlapping **5-minute temporal windows**:
$$\Delta t = 5 \text{ minutes} = 300 \text{ seconds} \implies 30 \text{ raw readings per window}$$

Aggregation reduced the dataset from **1,516,948 rows** to exactly **50,782 window instances**, a data compression ratio of approximately 30:1, while suppressing high-frequency sensor noise.

### 3.4 Feature Engineering (Mean, Std Dev, Min, Max $\rightarrow$ 64 Dimensions)
For each 5-minute window, four statistical summary moments were computed across all numeric channels:
1. **Window Mean ($\mu$)**: Captures the baseline central operating point (e.g., average motor current or average oil temperature during the window).
2. **Window Standard Deviation ($\sigma$)**: Captures mechanical fluctuation, vibration noise, and operational instability.
3. **Window Minimum ($\min$)**: Captures the deepest transient pressure drops or electrical troughs.
4. **Window Maximum ($\max$)**: Captures peak transient pressure surges or thermal peaks.

Across the 16 numerical dimensions, this transformation produced a rich, engineered feature matrix of:
$$\text{Total Feature Dimensions} = 16 \times 4 = 64 \text{ features}$$

### 3.5 Multivariate Proxy Anomaly Formulation (Median Absolute Deviation)
The MetroPT-3 dataset represents continuous unsupervised revenue telemetry without explicit, verified maintenance teardown failure logs. In industrial literature, applying supervised learning to unlabelled telemetry requires establishing a mathematically defensible **proxy anomaly ground truth**.

To achieve this without fabricating ground truth:
1. For each sensor channel $j$, the continuous median $\tilde{x}_j$ and Median Absolute Deviation ($\text{MAD}_j$) were calculated across the aggregated dataset:
   $$\text{MAD}_j = \text{median}\left( |x_{i,j} - \tilde{x}_j| \right)$$
2. Each window instance was transformed into a robust, scale-invariant deviation score:
   $$z_{i,j} = \frac{|x_{i,j} - \tilde{x}_j|}{\text{MAD}_j}$$
3. A composite multivariate anomaly score $S_i$ was computed as the average absolute deviation across all operational channels:
   $$S_i = \frac{1}{M}\sum_{j=1}^{M} z_{i,j}$$
4. The **98th percentile cutoff** ($Q_{0.98}$) of this score distribution was selected as the threshold boundary:
   $$y_i = \begin{cases} 1 & \text{if } S_i \ge \text{quantile}(S, 0.98) \\ 0 & \text{otherwise} \end{cases}$$

This isolates the top **2.00%** most extreme, abnormal multivariate operational regimes as anomalous proxy targets ($y=1$), leaving 98% as nominal operating conditions ($y=0$).

### 3.6 Severe Class Imbalance (98% Normal vs. 2% Anomaly Proxy)
The resulting target variable $y$ exhibits extreme industrial class imbalance:
- **Nominal Windows ($y=0$)**: $49,766$ windows ($98.00\%$)
- **Anomaly Proxy Windows ($y=1$)**: $1,016$ windows ($2.00\%$)
- **Total Windows ($N$)**: $50,782$ windows

This distribution mirrors real-world predictive maintenance, where healthy operational instances vastly outnumber failure or degradation periods.

### 3.7 Train / Validation / Test Stratified Partitioning Protocol
Because positive anomaly windows represent only 2% of the dataset, a naive chronological split carries severe risk: rare anomaly events may cluster in one seasonal period, leaving an evaluation partition with virtually zero anomalies.

To ensure statistically reliable evaluation across all five algorithms:
1. **Test Set Hold-Out (20%)**: A stratified random split separated **10,157 windows** (held out untouched for final evaluation).
2. **Training Partition (80%)**: **40,625 windows** were retained for model fitting and threshold selection.
3. **Internal Validation Split**: The 40,625 training windows were further partitioned into a fitting set (32,500 windows, 80%) and an internal validation set (8,125 windows, 20%) dedicated exclusively to decision threshold tuning.

---

# 4. ANALYSIS ON DATASET (CLASSIFIER COMPARATIVE BENCHMARKS)

To evaluate the predictive capability of classical machine learning on multivariate compressor telemetry, five distinct classifiers were benchmarked:
1. K-Nearest Neighbors (KNN)
2. Support Vector Machine (SVM)
3. Gaussian Naive Bayes (GNB)
4. Logistic Regression
5. Decision Tree

---

## 4.1 ANALYSIS 1: K-NEAREST NEIGHBORS (KNN)

### 4.1.i Introduction
K-Nearest Neighbors is a non-parametric, instance-based supervised classification algorithm. Rather than constructing an explicit mathematical decision boundary during training, KNN stores the standardized feature vectors and performs classification via local neighborhood voting during inference.

### 4.1.ii General Description & Algorithmic Background
Given an unlabelled query feature vector $\mathbf{x} \in \mathbb{R}^{64}$, KNN computes distance metrics between $\mathbf{x}$ and all stored training samples $\mathbf{x}_i \in \mathcal{D}_{\text{train}}$. The algorithm identifies the $k$ closest neighbors and assigns the posterior probability:
$$P(y=1|\mathbf{x}) = \frac{\sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i \cdot \mathbb{I}(y_i = 1)}{\sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i}$$
where $w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}$ represents inverse-distance weighting, ensuring closer neighbors exert greater voting influence than distant ones.

### 4.1.iii Specific Requirements, Hyperparameters & Formulas
- **Preprocessing Requirement**: StandardScaler is mandatory; distance computations are corrupted if features have disparate numerical scales.
- **Hyperparameter Configuration**:
  - Number of Neighbors ($k$): $7$
  - Metric: Euclidean distance: $d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{j=1}^{64} (u_j - v_j)^2}$
  - Weighting Strategy: Inverse distance (`weights="distance"`)
  - Validation-Tuned Threshold: $t^* = 0.5700$

### 4.1.iv Analysis Results
Evaluated on the 10,157 held-out test windows:

| Metric | Value | Evaluation Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | 0.9977 (99.77%) | High, but dominated by nominal class |
| **Precision** | 0.9688 (96.88%) | Very low false alarm rate (only 6 false positives) |
| **Recall (Sensitivity)** | 0.9163 (91.63%) | Successfully flagged 186 out of 203 anomalies |
| **$F_1$-Score (Test)** | **0.9418** | Harmonic balance between precision and recall |
| **Validation $F_1$** | 0.9571 | Stable generalization from validation tuning |
| **Balanced Accuracy** | 0.9578 | Arithmetic mean of sensitivity and specificity |
| **PR-AUC** | 0.9787 | Area under precision-recall curve |

**Confusion Matrix (KNN)**:
- True Negatives (TN): $9,948$
- False Positives (FP): $6$
- False Negatives (FN): $17$
- True Positives (TP): $186$

### 4.1.v Visualization & Sensitivity Dynamics
KNN exhibits tight, localized decision clusters. Because anomalous compressor windows form dense clusters in standardized multidimensional space, distance-weighted voting reliably isolates proxy anomalies with minimal false positives.

---

## 4.2 ANALYSIS 2: SUPPORT VECTOR MACHINE (SVM)

### 4.2.i Introduction
Support Vector Machine is a maximum-margin classification algorithm based on statistical learning theory. When data is non-linearly separable in the original 64-dimensional space, SVM maps input vectors into a high-dimensional Hilbert feature space where an optimal linear hyperplane can be constructed.

### 4.2.ii General Description & RBF Kernel Geometry
The SVM decision function is formulated as:
$$f(\mathbf{x}) = \text{sign}\left( \sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b \right)$$
Using the **Radial Basis Function (RBF)** kernel:
$$K(\mathbf{x}_i, \mathbf{x}) = \exp\left( -\gamma \|\mathbf{x}_i - \mathbf{x}\|^2 \right), \quad \gamma = \frac{1}{64 \cdot \sigma_X^2}$$
To counter the 98:2 class imbalance, the optimization penalty parameter $C$ is scaled inversely proportional to class frequencies:
$$C_1 = C \cdot \frac{N}{2 \cdot N_1}, \quad C_0 = C \cdot \frac{N}{2 \cdot N_0}$$

### 4.2.iii Specific Requirements, Hyperparameters & Formulas
- **Preprocessing Requirement**: Median Imputation + Standard Scaling.
- **Hyperparameter Configuration**:
  - Kernel: Radial Basis Function (RBF)
  - Regularization ($C$): $1.0$
  - Class Weight: Balanced
  - Probability Calibration: Enabled (Platt scaling)
  - Validation-Tuned Threshold: $t^* = 0.4724$

### 4.2.iv Analysis Results
Evaluated on the 10,157 held-out test windows:

| Metric | Value | Evaluation Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | 0.9980 (99.80%) | Superior overall classification rate |
| **Precision** | 0.9552 (95.52%) | 9 false alarms across 10,157 instances |
| **Recall (Sensitivity)** | 0.9458 (94.58%) | Detected 192 of 203 anomalous windows |
| **$F_1$-Score (Test)** | **0.9505** | Outstanding detection efficacy |
| **Validation $F_1$** | 0.9448 | Extremely tight validation-test agreement |
| **Balanced Accuracy** | 0.9725 | Superior balance across both classes |
| **PR-AUC** | **0.9826** | Highest PR-AUC among all five classifiers |

**Confusion Matrix (SVM)**:
- True Negatives (TN): $9,945$
- False Positives (FP): $9$
- False Negatives (FN): $11$
- True Positives (TP): $192$

### 4.2.v Visualization & Margin Behavior
The RBF kernel creates smooth, non-linear hyperspherical decision envelopes around multivariate compressor anomalies. The high PR-AUC (0.9826) demonstrates that SVM ranks anomalous windows with exceptional fidelity across varying threshold levels.

---

## 4.3 ANALYSIS 3: GAUSSIAN NAIVE BAYES (GNB)

### 4.3.i Introduction
Gaussian Naive Bayes is a generative probabilistic classifier derived from Bayes' Theorem, operating under the strong simplifying assumption that all feature dimensions are mutually conditionally independent given the class label.

### 4.3.ii General Description & Conditional Independence Principle
Under Bayes' theorem:
$$P(y=c|\mathbf{x}) = \frac{P(y=c) \prod_{j=1}^{64} P(x_j|y=c)}{P(\mathbf{x})}$$
The continuous likelihood of each feature is modeled as a normal distribution:
$$P(x_j|y=c) = \frac{1}{\sqrt{2\pi\sigma_{jc}^2}} \exp\left( -\frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2} \right)$$

### 4.3.iii Specific Requirements, Hyperparameters & Formulas
- **Preprocessing Requirement**: Standard scaling and variance smoothing ($\epsilon = 10^{-9}$).
- **Validation-Tuned Threshold**: $t^* = 1.0000$

### 4.3.iv Analysis Results
Evaluated on the 10,157 held-out test windows:

| Metric | Value | Evaluation Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | 0.9767 (97.67%) | Lowest accuracy among all models |
| **Precision** | **0.4612 (46.12%)**| Severe false alarm rate (236 false positives) |
| **Recall (Sensitivity)** | **0.9951 (99.51%)**| Flags 202 of 203 anomalies (near-perfect recall) |
| **$F_1$-Score (Test)** | **0.6303** | Severely compromised by low precision |
| **Validation $F_1$** | 0.6468 | Confirms poor precision on validation set |
| **Balanced Accuracy** | 0.9857 | High due to extreme recall |
| **PR-AUC** | 0.4590 | Severely degraded precision-recall envelope |

**Confusion Matrix (GNB)**:
- True Negatives (TN): $9,718$
- False Positives (FP): $236$
- False Negatives (FN): $1$
- True Positives (TP): $202$

### 4.3.v Visualization & False Alarm Manifestation
The performance of Gaussian Naive Bayes illustrates a foundational machine learning pathology: **violation of conditional independence**. In industrial compressors, physical variables are heavily correlated (e.g., motor current and delivery pressure rise simultaneously; ambient temperature tracks discharge temperature). Because GNB multiplies these correlated probabilities, it drastically overestimates posterior anomaly probabilities, triggering 236 false alarms and rendering the model impractical for real-world operations despite capturing 99.5% of anomalies.

---

## 4.4 ANALYSIS 4: LOGISTIC REGRESSION (OPTIMAL CLASSIFIER)

### 4.4.i Introduction
Logistic Regression is a discriminative linear classification model that models the posterior probability of the positive class via the logistic sigmoid function operating on a linear combination of input features.

### 4.4.ii General Description & Log-Odds Optimization
The model parameters $\mathbf{w} \in \mathbb{R}^{64}$ and bias $b$ are learned by minimizing the binary cross-entropy loss function with $L_2$ Tikhonov regularization:
$$\min_{\mathbf{w}, b} \left[ \frac{1}{2}\|\mathbf{w}\|_2^2 + C \sum_{i=1}^{N_{\text{train}}} w_{y_i} \log\left( 1 + \exp\left( -y_i (\mathbf{w}^T \mathbf{x}_i + b) \right) \right) \right]$$
The posterior probability score is generated via the sigmoid mapping:
$$\hat{p}(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + \exp(-(\mathbf{w}^T \mathbf{x} + b))}$$

### 4.4.iii Specific Requirements, Regularization & Threshold Tuning
- **Preprocessing Requirement**: StandardScaler and median imputation.
- **Hyperparameter Configuration**:
  - Regularization Penalty: $L_2$
  - Inverse Regularization Strength ($C$): $1.0$
  - Solver: L-BFGS (`max_iter=2000`)
  - Class Weighting: Balanced
  - **Validation-Tuned Decision Threshold**: $t^* = 0.948061$

**Threshold Optimization Methodology**: Rather than relying on the naive default threshold $t = 0.5$, threshold candidates were evaluated on the internal validation set across 50 quantiles ($[0.50, 0.99]$). The threshold maximizing validation $F_1$ was identified at $t^* = 0.948061$.

### 4.4.iv Analysis Results
Evaluated on the 10,157 held-out test windows:

| Metric | Value | Evaluation Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **0.9981 (99.81%)**| Highest accuracy among all classifiers |
| **Precision** | 0.9466 (94.66%) | Only 11 false positives |
| **Recall (Sensitivity)** | **0.9606 (96.06%)**| Successfully detected 195 of 203 anomalies |
| **$F_1$-Score (Test)** | **0.9535** | **Highest test $F_1$-score among all 5 models** |
| **Validation $F_1$** | **0.9632** | Highest validation $F_1$-score |
| **Balanced Accuracy** | **0.9797** | Exceptional balanced performance |
| **PR-AUC** | 0.9798 | Strong precision-recall area |

**Confusion Matrix (Logistic Regression)**:
- True Negatives (TN): $9,943$
- False Positives (FP): $11$
- False Negatives (FN): $8$
- True Positives (TP): $195$

### 4.4.v Visualization & Score vs. Threshold Dynamics
Combining standardized multivariate inputs with balanced class weighting and optimal decision thresholding ($t^* = 0.948061$) allowed Logistic Regression to construct an exceptionally stable linear separator. The model generated only 8 false negatives and 11 false alarms across more than 10,000 instances.

---

## 4.5 ANALYSIS 5: DECISION TREE CLASSIFIER

### 4.5.i Introduction
A Decision Tree is a non-parametric model that partitions feature space into orthogonal hyper-rectangles using recursive binary splitting. It provides high operational interpretability through explicit if-then rules.

### 4.5.ii General Description & Recursive Binary Splitting
At each internal node $m$, the algorithm scans all 64 candidate features $j$ and split thresholds $\theta$ to maximize Gini impurity reduction:
$$\Delta I(m) = I(m) - \left( \frac{N_L}{N_m} I(m_L) + \frac{N_R}{N_m} I(m_R) \right)$$
where the Gini impurity is defined as:
$$I(m) = 1 - \sum_{c \in \{0, 1\}} p_{mc}^2$$

### 4.5.iii Specific Requirements, Hyperparameters & Formulas
- **Hyperparameter Configuration**:
  - Splitting Criterion: Gini impurity
  - Maximum Tree Depth (`max_depth`): $6$ (prevents memorization of noise)
  - Minimum Samples per Leaf (`min_samples_leaf`): $10$
  - Class Weight: Balanced
  - Validation-Tuned Threshold: $t^* = 0.9979$

### 4.5.iv Analysis Results
Evaluated on the 10,157 held-out test windows:

| Metric | Value | Evaluation Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | 0.9967 (99.67%) | High accuracy |
| **Precision** | 0.8967 (89.67%) | 22 false alarms |
| **Recall (Sensitivity)** | 0.9409 (94.09%) | Detected 191 of 203 anomalies |
| **$F_1$-Score (Test)** | **0.9183** | Solid performance, below linear/kernel models |
| **Validation $F_1$** | 0.9271 | High consistency between validation and test |
| **Balanced Accuracy** | 0.9693 | Strong balanced detection |
| **PR-AUC** | 0.8791 | Lowest PR-AUC among non-naive classifiers |

**Confusion Matrix (Decision Tree)**:
- True Negatives (TN): $9,932$
- False Positives (FP): $22$
- False Negatives (FN): $12$
- True Positives (TP): $191$

### 4.5.v Visualization & Split Hierarchy
The top branching nodes of the decision tree selected standard deviations of `Motor_current`, peak discharge pressure `TP3_max`, and oil temperature deviations `Oil_temperature_std`. While interpretable, the axis-aligned splits of a single tree exhibit higher variance compared to continuous linear or kernel boundaries.

---

## 4.6 CROSS-MODEL COMPARATIVE SYNTHESIS & BENCHMARK DISCUSSION

### 4.6.1 Leaderboard Ranking
The comprehensive comparative performance across all five algorithms on the identical held-out test set ($N=10,157$) is summarized below:

| Rank | Classifier Algorithm | Test $F_1$ | Validation $F_1$ | PR-AUC | Precision | Recall | Balanced Acc | Test Accuracy | Decision Threshold |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Logistic Regression** | **0.9535** | **0.9632** | 0.9798 | 0.9466 | **0.9606** | **0.9797** | **0.9981** | 0.948061 |
| **2** | **Support Vector Machine (SVM)** | 0.9505 | 0.9448 | **0.9826** | 0.9552 | 0.9458 | 0.9725 | 0.9980 | 0.472386 |
| **3** | **K-Nearest Neighbors (KNN)** | 0.9418 | 0.9571 | 0.9787 | **0.9688** | 0.9163 | 0.9578 | 0.9977 | 0.569997 |
| **4** | **Decision Tree** | 0.9183 | 0.9271 | 0.8791 | 0.8967 | 0.9409 | 0.9693 | 0.9967 | 0.997885 |
| **5** | **Gaussian Naive Bayes** | 0.6303 | 0.6468 | 0.4590 | 0.4612 | 0.9951 | 0.9857 | 0.9767 | 1.000000 |

### 4.6.2 Confusion Matrix Comparative Matrix
The diagnostic breakdown across all 10,157 test instances demonstrates the real-world operational implications of each classifier:

| Classifier | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | False Alarm Rate | Missed Failure Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 9,943 | 11 | **8** | 195 | 0.11% | **3.94%** |
| **SVM** | 9,945 | 9 | 11 | 192 | 0.09% | 5.42% |
| **KNN** | **9,948** | **6** | 17 | 186 | **0.06%** | 8.37% |
| **Decision Tree** | 9,932 | 22 | 12 | 191 | 0.22% | 5.91% |
| **Gaussian Naive Bayes** | 9,718 | 236 | **1** | **202** | 2.37% | **0.49%** |

### 4.6.3 Why Accuracy Fails: The 98% Base-Rate Fallacy
A critical finding of this study is the complete inadequacy of standard **Accuracy** as an evaluation metric in predictive maintenance.
- A trivial dummy baseline predicting $y=0$ (Normal) 100% of the time achieves **98.00% accuracy**.
- However, its recall is **0.00%**, missing every single equipment breakdown.
- While Gaussian Naive Bayes scored 97.67% accuracy, it flooded the maintenance queue with 236 false alarms.
- Therefore, $F_1$-score (harmonic mean of precision and recall) and PR-AUC serve as the only reliable academic standards for rare industrial failure detection.

### 4.6.4 Algorithmic Rationale: Why Logistic Regression Won
Logistic Regression attained the highest test $F_1$ (0.9535) and validation $F_1$ (0.9632) due to three key factors:
1. **Multi-Feature Co-linearity Alignment**: By standardizing the 64 features and tuning $L_2$ regularization, correlated sensor pairs (pressure, temperature, current) contributed linearly without inflating posterior variance.
2. **Effective Class Weighting**: Weighting training samples inversely to class prevalence balanced the log-odds gradient descent.
3. **Validation Decision Thresholding**: Calibrating the cutoff to $t^* = 0.948061$ on the validation set successfully mitigated false alarms while retaining 96.06% sensitivity.

---

# 5. CONCLUSION

### 5.1 Project Summary
This project successfully designed, engineered, and benchmarked an end-to-end predictive maintenance pipeline for urban rail transit air compressors utilizing the 1.5-million-row UCI MetroPT-3 dataset. Key deliverables achieved:
1. **Engineered Preprocessing Pipeline**: Converted high-frequency 10-second logs into 50,782 5-minute statistical windows (64 dimensions), boosting data ingestion speed by over 300x through persistent Parquet caching.
2. **Transparent Proxy Labeling**: Formulated a mathematically sound multivariate anomaly proxy via Median Absolute Deviation at the 98th percentile, replicating the real-world 98:2 operational class imbalance.
3. **Five-Classifier Academic Benchmark**: Successfully implemented and rigorously compared KNN, SVM, Gaussian Naive Bayes, Logistic Regression, and Decision Tree on held-out test data.
4. **Optimal Architecture Identification**: Identified Logistic Regression with validation threshold tuning ($t^* = 0.948061$) as the top-performing model, achieving **$F_1 = 0.9535$**, **Recall = 0.9606**, **Precision = 0.9466**, and **PR-AUC = 0.9798**.
5. **Interactive Industrial AI Dashboard**: Delivered a responsive, modern Streamlit dashboard supporting sub-second multi-channel telemetry visualization, threshold gauge inspection, and historical window inference.

### 5.2 Key Technical Takeaways
- Distance and margin classifiers (KNN, SVM) perform exceptionally well when standard scaling is applied, with SVM achieving the top PR-AUC (0.9826).
- Generative probabilistic models (Naive Bayes) fail in multi-sensor physical systems due to correlated telemetry channels, producing unacceptable false alarm rates (236 false positives).
- Validation-based threshold tuning is essential for handling severe class imbalance without contaminating held-out test data.

### 5.3 Methodological Integrity Regarding Proxy Labels
A core contribution of this work is scientific transparency. Because public telemetry lacks verified component teardown records, model outputs are strictly categorized as **statistical operational anomaly proxies** rather than certified mechanical breakdowns. This prevents the dangerous overclaiming of predictive accuracy in safety-critical rail systems.

---

# 6. FUTURE SCOPE

To evolve this academic prototype into an operational enterprise predictive maintenance solution, several research avenues are identified:

### 6.1 Integration of Certified Run-to-Failure Physical Logs
Future research should partner with transit maintenance depots to ingest maintenance ticketing systems, work order logs, and component replacement timestamps. Correlating statistical telemetry spikes with actual physical component degradation (e.g., valve spring fatigue, desiccant saturation, bearing spalling) will enable true Remaining Useful Life (RUL) regression modeling.

### 6.2 Deep Temporal Architectures (LSTM, GRU, TCN)
While classical machine learning on statistical window features proved highly effective ($F_1 > 0.95$), deep sequential neural networks can capture multi-step temporal dependencies directly from raw time series:
- **Long Short-Term Memory (LSTM)** and **Gated Recurrent Units (GRU)** for sequential memory.
- **Temporal Convolutional Networks (TCN)** with dilated causal convolutions for large receptive fields.
- **Transformer-based Self-Attention** for multi-head sensor cross-correlation.

### 6.3 Edge SCADA Integration & Industrial Stream Processing
Deploying models onto locomotive edge hardware (e.g., NVIDIA Jetson or industrial ARM gateways) via lightweight runtimes (ONNX / TensorRT). Telemetry can be ingested directly via industrial **MQTT** or **OPC-UA** protocols for sub-second anomaly alerting to locomotive drivers and central dispatchers.

### 6.4 Explainable AI (XAI) for Maintenance Crews
Integrating **SHAP (SHapley Additive exPlanations)** and **Integrated Gradients** into the inspection dashboard to generate localized feature attribution explanations. Providing technicians with plain-language diagnostic summaries (e.g., *"Flagged due to a 3.4σ spike in Motor Current combined with a 2.8σ drop in Reservoirs Pressure"*) bridges the gap between machine learning models and mechanical maintenance teams.

---

# 7. REFERENCES

1. **Veloso, B. M., Gama, J., Malheiro, B., & Burguillo, J. C. (2022)**. "A Real-World Dataset for Multi-Stage Predictive Maintenance in Metro Trains." *Sensors*, 22(8), 2955. DOI: `10.3390/s22082955`.
2. **UCI Machine Learning Repository (2022)**. "MetroPT-3 Dataset: Air Compressor Telemetry." University of California, Irvine, School of Information and Computer Sciences. URI: `https://archive.ics.uci.edu/dataset/791/metropt3%2Bdataset`.
3. **Mobley, R. K. (2002)**. *An Introduction to Predictive Maintenance*. 2nd Edition, Elsevier Butterworth-Heinemann, Oxford.
4. **Jardine, A. K., Lin, D., & Banjevic, D. (2006)**. "A review on machinery diagnostics and prognostics implementing condition-based maintenance." *Mechanical Systems and Signal Processing*, 20(7), 1483-1510.
5. **Pedregosa, F., et al. (2011)**. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825-2830.
6. **Saito, T., & Rehmsmeier, M. (2015)**. "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Imbalanced Datasets." *PLOS ONE*, 10(3), e0118432.
7. **Hastie, T., Tibshirani, R., & Friedman, J. (2009)**. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd Edition, Springer-Verlag, New York.
8. **He, H., & Garcia, E. A. (2009)**. "Learning from Imbalanced Data." *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.
9. **Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013)**. "Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median." *Journal of Experimental Social Psychology*, 49(4), 764-766.
10. **ISO 13381-1:2015**. "Condition monitoring and diagnostics of machines — Prognostics — Part 1: General guidelines." International Organization for Standardization, Geneva.
