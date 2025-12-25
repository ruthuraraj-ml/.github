# 🚀 ruthuraraj-ml — Machine Learning & Generative AI Projects

Welcome to **ruthuraraj-ml**, an organization showcasing the end-to-end Machine Learning,  
Deep Learning, and Generative AI projects developed by **R. Ruthuraraj (AP/Mechanical Engg.)**  
as part of continued learning, QIP coursework, teaching, and applied AI research.

This portfolio includes classical ML, DNNs, GANs, multimodal AI, and domain-specific  
engineering applications — each packaged with reports, notebooks, reproducible code, and documentation.

---

# 📚 Project Index

Below is the complete list of projects currently published in this organization.  
Each repository includes:  
✔ Report  
✔ Colab notebook  
✔ README  
✔ Requirements  
✔ Folder structure  
✔ License  
✔ Reproducible workflow

---

## 🔍 **1. Advertising Sales Prediction — Linear Regression**
**Type:** Regression  
Predicts product sales using TV, Radio, and Newspaper ad spending.  
Includes EDA, correlation analysis, multicollinearity checks, and regression diagnostics.  
**Models:** Linear Regression  
🔗 Repository: `[Advertising-Sales-Prediction](https://github.com/ruthuraraj-ml/Advertising-Sales-Prediction-using-Linear-Regression)`

---

## 🚲 **2. Bike Sharing Demand Prediction — Linear Regression**
**Type:** Time-Pattern Regression  
Analyses hourly rental counts using weather, season, and time-based features.  
Highlights temporal patterns, weather effects, and demand forecasting.  
**Models:** Linear Regression  
🔗 Repository: `[Bike-Demand-Prediction](https://github.com/ruthuraraj-ml/Bike-Sharing-Demand-Prediction)`

---

## 🩺 **3. Diabetes Prediction — Logistic Regression**
**Type:** Medical Classification  
Binary diabetes prediction using clinical features from the Pima Indians Dataset.  
Includes preprocessing, correlation, model metrics, ROC–AUC, and interpretability.  
**Models:** Logistic Regression  
🔗 Repository: `[Diabetes-Prediction](https://github.com/ruthuraraj-ml/Diabetes-Prediction-using-Logistic-Regression)`

---

## 🚢 **4. Titanic Survival Prediction — Logistic Regression**
**Type:** Binary Classification  
Predicts survival probabilities using demographics, ticket class, family size, fare, etc.  
Includes categorical encoding, scaling, and model evaluation metrics.  
**Models:** Logistic Regression  
🔗 Repository: `[Titanic-Survival-Prediction](https://github.com/ruthuraraj-ml/Titanic-Survival-Prediction-using-Logistic-Regression)`

---

## 🍷 **5. Wine Quality Prediction — Random Forest Classifier**
**Type:** Multiclass Classification  
Predicts wine quality (3–8) from physicochemical attributes using ensemble learning.  
Includes EDA, class imbalance analysis, feature importance, cross-validation.  
**Models:** Random Forest  
🔗 Repository: `[Wine-Quality-Prediction](https://github.com/ruthuraraj-ml/Wine-Quality-Prediction-using-Random-Forest-Classifier)`

---

## 💳 **6. Online Payment Fraud Detection — ML (LR, RF, XGBoost)**
**Type:** Imbalanced Classification  
Detects fraudulent transactions in extreme class imbalance (~0.15%).  
Compares Logistic Regression, Random Forest, and XGBoost.  
Includes PR–AUC, ROC–AUC, and confusion matrices.  
**Models:** LR, RF, XGBoost  
🔗 Repository: `[Fraud-Detection-ML](https://github.com/ruthuraraj-ml/Online-Payment-Fraud-Detection-using-Machine-Learning)`

---

## 🖼️ **7. Image Caption Generator — Vision–Language Models (Generative AI)**

**Type:** Multimodal AI (Vision + Language)

This project explores **automatic image caption generation** using both **classical encoder–decoder architectures** and **modern transformer-based vision–language models**.

The work begins with a **CNN–LSTM baseline** (InceptionV3 + LSTM) to validate cross-modal learning, followed by experiments with a **Transformer decoder** and **pretrained vision–language transformers** fine-tuned on a small real-world dataset.

Rather than focusing only on accuracy, the project emphasizes:
- model validation via overfitting tests,
- the effect of dataset quality on caption fluency,
- and practical challenges of fine-tuning multimodal models on limited data and hardware.

**Key Components**
- CNN encoder (InceptionV3, ImageNet pretrained)
- LSTM / Transformer-based caption decoder
- Pretrained vision–language transformer fine-tuning
- Caption cleaning and preprocessing pipeline
- Beam search, repetition penalty, and decoding strategies
- Qualitative and empirical analysis of fine-tuning behavior

**Datasets**
- Open Images Captions (Micro) — Hugging Face

**Models Explored**
- CNN + LSTM (Encoder–Decoder)
- CNN + Transformer Decoder (from scratch)
- Pretrained Vision–Language Transformer (fine-tuned)

**Outcome**
- Demonstrates correct multimodal learning via controlled overfitting
- Shows why pretrained captioning models outperform naive fine-tuning on small datasets
- Serves as a **learning-focused Generative AI case study**, not just a benchmark-chasing implementation
  
🔗 Repository: `[Image-Caption-Generator](https://github.com/ruthuraraj-ml/Image-Caption-Generation-using-Vision-Language-Models))`

---

## 🔍🖼️ **8. Image Super-Resolution — SRGAN & ESRGAN**

**Type:** Generative Adversarial Networks (Perceptual Super-Resolution)  

Reconstructs ×4 high-resolution images from low-resolution inputs using **SRGAN** and its improved variant **ESRGAN (RRDB-based)**.

Includes DIV2K dataset, patch-based training, warm-up and stability-oriented adversarial fine-tuning, and full-image qualitative evaluation.

Emphasizes perceptual trade-offs under constrained training, with side-by-side and zoomed visual comparisons.

**Models:** SRGAN, ESRGAN (Generator + Discriminator) 

🔗 Repository: `[SRGAN, ESRGAN-SuperResolution](https://github.com/ruthuraraj-ml/Image-Super-Resolution-using-SRGAN-and-ESRGAN)`

---

# 📌 Upcoming Projects

Projects planned for future upload:
 
- 📷 **ESRGAN / Real-ESRGAN extension**  
- 🤖 **VAE for Tabular Data (Generative Modeling for ML datasets)**  
- 📄 **RAG-based LMS System (Auto notes + question generation)**  
- 📈 **Surface Roughness Prediction (Mechanical Engineering + ML)**  
- 🛠️ **Nano-additives ML Modelling (Bio-lubricants)**  

---

# 🧱 Technology Stack Overview

- **Languages:** Python  
- **ML Libraries:** scikit-learn, NumPy, Pandas  
- **DL Libraries:** PyTorch, TensorFlow/Keras  
- **GenAI:** CNN–LSTM, SRGAN, VGG Perceptual Loss  
- **Tools:** Google Colab, HuggingFace, Jupyter Notebooks  
- **Domains:** Regression, Classification, Ensemble Learning, GANs, Multimodal AI  

---

# 👨‍🏫 About

**R. Ruthuraraj**  
Assistant Professor (Mechanical Engineering)  
Specializing in Machine Learning, Deep Learning, Generative AI, and AI-enabled Engineering Applications.

This organization represents his journey from classical ML → modern Deep Learning → Generative AI.

---

# ⭐ Acknowledgements

- AICTE QIP Programme on 'AI to Generative AI' — IIIT Allahabad
- SNS College of Technology
- Kaggle, UCI, HuggingFace Datasets  
- PyTorch & TensorFlow communities  
- SRGAN / ESRGAN research papers  

---

⭐ *If you find the projects useful, please star the repositories!*  
