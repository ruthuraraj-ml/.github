# 🚀 ruthuraraj-ml — Machine Learning & Generative AI Projects

> *This portfolio reflects a structured learning journey from **Classical Machine Learning → Deep Learning → Representation Learning → Generative AI → Agentic AI Systems**.*

Welcome to **ruthuraraj-ml**, an organization showcasing the end-to-end Machine Learning,  
Deep Learning, and Generative AI projects developed by **R. Ruthuraraj (AP/Mechanical Engg.)**  
as part of continued learning, QIP coursework, teaching, and applied AI research.

This portfolio includes classical ML, DNNs, GANs, multimodal AI, and domain-specific  
engineering applications — each packaged with reports, notebooks, reproducible code, and documentation.

---

# 🗺️ Portfolio Structure

Projects in this organization are organized to reflect the **evolution of modern AI methods**:

Classical Machine Learning

        ↓
Neural Networks & Deep Learning

        ↓
Representation Learning

        ↓
Agentic AI Systems

        ↓
Generative & Multimodal AI

        ↓
AI for Engineering Applications


This structure mirrors the conceptual progression from **statistical learning → neural representation learning → generative models → intelligent AI systems**.

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

## 🧠 Foundations & Core Concepts

### 🔹 1. XOR Problem — Why Deep Learning Exists
**Type:** Learning Theory / Neural Network Foundations  

A minimal, concept-driven demonstration explaining **why linear models and logistic regression fail**, and **why neural networks with hidden layers are necessary**.

This project progresses through:
- OR and AND problems (linear separability)
- XOR problem (linear inseparability)
- Logistic regression failure on XOR
- Successful solution using a neural network with one hidden layer

The focus is on **decision boundaries, representation learning, and architectural necessity**, making this a conceptual bridge between classical machine learning and deep learning.

**Models:** Logistic Regression, Multi-Layer Perceptron (MLP)

🔗 Repository: `[XOR-Why-Deep-Learning-Exists](https://github.com/ruthuraraj-ml/XOR-Why-Deep-Learning-Exists)`

---

### 🔹 2. Neural Networks — From Basics to Stabilization (PyTorch)
**Type:** Neural Network Foundations / Training Dynamics  

A teaching-oriented, hands-on walkthrough of neural networks using PyTorch, designed to explain **why key neural network components exist and how they affect learning behavior**.

This project incrementally builds neural networks and studies the effect of:
- network depth and fully connected architectures,
- Batch Normalization for training stability,
- different activation functions,
- optimizer choices and convergence behavior,
- Dropout-based regularization,
- inference on unseen real-world data,
- extension from binary to multiclass classification.

The emphasis is on **training dynamics, representation learning, and stabilization techniques**, rather than benchmark optimization.

**Models:** Feedforward Neural Networks (MLP), BatchNorm Networks, Dropout Networks  

🔗 Repository: `[Neural-Networks-Demo-PyTorch]([https://github.com/ruthuraraj-ml/Neural-Networks-Demo-PyTorch])`

---

## 📐 Classical Machine Learning (Supervised Learning)

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

## 🔤 Embeddings & Representation Learning

### 🔹 Word2Vec Embedding Explorer — From Training to Semantic Geometry
**Type:** Representation Learning / NLP Foundations  

An end-to-end implementation of Word2Vec (Skip-Gram with Negative Sampling) built from scratch in PyTorch and extended into a fully interactive browser-based embedding explorer.

This project does not stop at training embeddings — it exposes how semantic structure emerges during learning by exporting intermediate checkpoints and visualizing them interactively.

The system trains embeddings on the WikiText-2 corpus, exports selected vocabulary vectors across epochs, and enables live exploration of the vector space including nearest neighbors, similarity scoring, and analogy solving.

**Concepts demonstrated:**
- distributional hypothesis (“meaning from context”)
- negative sampling and subsampling
- vector similarity (cosine distance)
- semantic clustering
- word analogies as vector arithmetic
- geometry of learned representations
- effect of training progression on embedding quality

The accompanying interactive web demo allows observing how random vectors gradually organize into meaningful semantic groups.

**Models:** Word2Vec (Skip-Gram), Embedding Space Visualization  

🔗 Repository: `[Embedding_Search](https://github.com/ruthuraraj-ml/Embedding_Search)`

🔗 Live Demo: `https://ruthuraraj-ml.github.io/Embedding_Search/`

---

## 🤖🧩 Agentic AI & LLM Systems

### 🔹 AI Content Studio — Multi-Agent System for Educational Content Generation

**Type:** Agentic AI / LLM Systems / AI Automation  

An experimental **multi-agent AI system** that transforms a simple topic prompt into structured educational content by orchestrating specialized AI agents.

Instead of relying on a single LLM, the system coordinates a **team of task-specific agents**, each responsible for a different stage of content creation — similar to a real production pipeline.

The system includes:

- **Research Agent** – retrieves factual information using Retrieval-Augmented Generation (RAG)  
- **Image Agent** – converts conceptual explanations into scientific diagram prompts and generates diagrams using diffusion models  
- **Reviewer Agent** – refines explanations into structured, beginner-friendly teaching content  
- **Manager Agent** – orchestrates the workflow and coordinates agent collaboration

This architecture demonstrates how **complex tasks can be decomposed into cooperative AI agents**, illustrating the emerging paradigm of **Agentic AI systems built on top of large language models**.

---

### Key Concepts Demonstrated
- Multi-agent orchestration
- Retrieval-Augmented Generation (RAG)
- Tool-using LLM agents
- Diagram generation with diffusion models
- Structured explanation generation for teaching
- Streamlit-based interactive AI interface

---

### Tech Stack
- **CrewAI** — multi-agent orchestration  
- **LLM APIs** — Gemini / Groq  
- **HuggingFace Diffusion Models** — FLUX image generation  
- **RAG Pipeline** — knowledge retrieval system  
- **Streamlit** — interactive demo interface  

---

### Outcome
- Demonstrates how **LLM-based agents can collaborate to produce multimodal educational content**
- Serves as a practical introduction to **Agentic AI architectures**

🔗 Repository: `[AI-Content-Studio-Multi-Agent-System](https://github.com/ruthuraraj-ml/Multi-Agent-AI-System-for-Educational-Content-Generation)`

---

## 🧠🤖 Generative & Multimodal AI

## 🖼️ **1. Image Caption Generator — Vision–Language Models (Generative AI)**

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

## 🔍🖼️ **2. Image Super-Resolution — SRGAN & ESRGAN**

**Type:** Generative Adversarial Networks (Perceptual Super-Resolution)  

Reconstructs ×4 high-resolution images from low-resolution inputs using **SRGAN** and its improved variant **ESRGAN (RRDB-based)**.

Includes DIV2K dataset, patch-based training, warm-up and stability-oriented adversarial fine-tuning, and full-image qualitative evaluation.

Emphasizes perceptual trade-offs under constrained training, with side-by-side and zoomed visual comparisons.

**Models:** SRGAN, ESRGAN (Generator + Discriminator) 

🔗 Repository: `[SRGAN, ESRGAN-SuperResolution](https://github.com/ruthuraraj-ml/Image-Super-Resolution-using-SRGAN-and-ESRGAN)`

---

## 🚧 Research & Development Roadmap

The following projects are currently under development and reflect the long-term direction of this profile — bridging **representation learning, generative AI, and engineering applications**.

### 🧠 Generative AI & Representation Learning
- **Super-Resolution with ESRGAN / Real-ESRGAN Extension**  
  Training and fine-tuning image super-resolution models with domain-specific datasets and evaluation of perceptual quality metrics.

- **Variational Autoencoder (VAE) for Tabular Data**  
  Generating synthetic ML datasets and studying latent structure learning for structured data (non-image generative modeling).

- **Retrieval-Augmented Generation (RAG) Learning Management System**  
  Automatic lecture note generation, semantic search, and question creation from educational content.

---

### ⚙️ AI for Mechanical & Manufacturing Engineering
- **Surface Roughness Prediction using ML/DL**  
  Predicting machining quality metrics from cutting parameters and process conditions.

- **Nano-Additive Bio-Lubricant Modeling**  
  Machine learning modeling of tribological performance and optimization of eco-friendly lubricant compositions.

---

### 🎯 Research Direction
> Building interpretable AI systems that connect **mathematical learning principles → real engineering decision making**.  

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
