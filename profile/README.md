# 🚀 ruthuraraj-ml — Machine Learning & Generative AI Portfolio

> *A structured learning journey: **Classical ML → Deep Learning → Representation Learning → Generative AI → Agentic AI Systems***

**R. Ruthuraraj** · Assistant Professor, Mechanical Engineering · SNS College of Technology  
*QIP Programme on 'AI to Generative AI' — IIIT Allahabad*

---

## 💡 Philosophy

> **These projects are not about replicating state-of-the-art benchmarks or shipping polished end products.**

The goal of every repository here is to **build from scratch after genuinely understanding the underlying concepts** — and then to be honest about where things break, what the model cannot do, and why. Limitations are documented as carefully as results, because understanding failure modes is how real learning happens.

This matters especially coming from a Mechanical Engineering background: the instinct here is not to chase accuracy numbers, but to ask *why does this architecture work, where does it fail, and what does that tell us about the problem?* That question drives every project in this portfolio.

If you are a student, researcher, or practitioner looking for polished, production-ready implementations — these may not be what you need. 

If you are trying to **actually understand** how these systems work from the ground up, you are in the right place.

---

## 🗺️ Learning Progression

```
Classical Machine Learning  →  Neural Networks & Deep Learning
        ↓                               ↓
Representation Learning     →  Generative & Multimodal AI
                                       ↓
              Agentic AI Systems & LLM Applications
                            ↓
              AI for Engineering Applications
```

Each project is packaged with a **report, notebook, README, requirements, and reproducible workflow**.

---

## 🧠 Foundations & Core Concepts

### [XOR Problem — Why Deep Learning Exists](https://github.com/ruthuraraj-ml/XOR-Why-Deep-Learning-Exists)
![Type](https://img.shields.io/badge/Type-Learning%20Theory-blue) ![Framework](https://img.shields.io/badge/Framework-scikit--learn-orange)

A concept-driven demonstration of **why linear models fail and why hidden layers are necessary**. The project deliberately walks through OR and AND problems (linearly separable), then breaks logistic regression on XOR to show where it fails and why — before solving it with a single hidden layer MLP. The focus is entirely on decision boundaries and architectural necessity, not accuracy. 

This is a conceptual bridge between classical ML and deep learning, not a benchmark exercise.

`Logistic Regression` `MLP` `Decision Boundaries`

---

### [Neural Networks — From Basics to Stabilization (PyTorch)](https://github.com/ruthuraraj-ml/Neural-Networks-Demo-PyTorch)
![Type](https://img.shields.io/badge/Type-Training%20Dynamics-blue) ![Framework](https://img.shields.io/badge/Framework-PyTorch-red)

A teaching-oriented walkthrough that incrementally builds networks and isolates the effect of one component at a time — BatchNorm, activation functions, optimizers, Dropout, and multiclass extension. The emphasis is on *why* each component was invented and what training instability looks like without it, rather than achieving a target metric. Limitations at each stage are explicitly shown before the fix is introduced.

`PyTorch` `BatchNorm` `Dropout` `Training Dynamics`

---

## 📐 Classical Machine Learning

| # | Project | Type | Models |
|---|---------|------|--------|
| 1 | [Advertising Sales Prediction](https://github.com/ruthuraraj-ml/Advertising-Sales-Prediction-using-Linear-Regression) | Regression | Linear Regression |
| 2 | [Bike Sharing Demand Prediction](https://github.com/ruthuraraj-ml/Bike-Sharing-Demand-Prediction) | Time-Pattern Regression | Linear Regression |
| 3 | [Diabetes Prediction](https://github.com/ruthuraraj-ml/Diabetes-Prediction-using-Logistic-Regression) | Medical Classification | Logistic Regression |
| 4 | [Titanic Survival Prediction](https://github.com/ruthuraraj-ml/Titanic-Survival-Prediction-using-Logistic-Regression) | Binary Classification | Logistic Regression |
| 5 | [Wine Quality Prediction](https://github.com/ruthuraraj-ml/Wine-Quality-Prediction-using-Random-Forest-Classifier) | Multiclass Classification | Random Forest |
| 6 | [Online Payment Fraud Detection](https://github.com/ruthuraraj-ml/Online-Payment-Fraud-Detection-using-Machine-Learning) | Imbalanced Classification | LR · RF · XGBoost |

---

## 🔤 Embeddings & Representation Learning

### [Word2Vec Embedding Explorer](https://github.com/ruthuraraj-ml/Embedding_Search) · [🌐 Live Demo](https://ruthuraraj-ml.github.io/Embedding_Search/)
![Type](https://img.shields.io/badge/Type-Representation%20Learning-purple) ![Framework](https://img.shields.io/badge/Framework-PyTorch-red)

End-to-end Word2Vec (Skip-Gram + Negative Sampling) built from scratch in PyTorch, extended into an **interactive browser-based embedding explorer**. Rather than stopping at training, the project exports intermediate checkpoints across epochs and visualises how semantic structure gradually emerges from random vectors — nearest neighbours, similarity scoring, analogy solving, and geometric clustering. The live demo makes the abstract distributional hypothesis tangible: you can watch meaning form in real time.

**Concepts covered:** distributional hypothesis · negative sampling · cosine similarity · semantic clustering · vector arithmetic · geometry of learned representations · effect of training progression on embedding quality

`Word2Vec` `Skip-Gram` `FAISS` `Embedding Visualisation` `WikiText-2`

---

## 🤖 Agentic AI & LLM Systems

### [Workshop Assistant — ReAct-lite RAG Agent](https://github.com/ruthuraraj-ml/Workshop-Assistant-RAG-Agent)
![Type](https://img.shields.io/badge/Type-RAG%20%7C%20Agentic%20AI-darkgreen) ![Built](https://img.shields.io/badge/Built%20For-Live%20Workshop%20Demo-yellow)

Built as a **live demo for the final session** of a 3-day AI workshop (*From Machine Learning to AI Agents*, SNS College of Technology, Jan 2026). The agent answers questions grounded in workshop PDFs and notebooks, using a heuristic ReAct-style decision loop. 

The core design decision — replacing an LLM routing call with a **FAISS L2 distance threshold** — was deliberate: it eliminates one API call per query, is fully deterministic, and makes the agent's reasoning transparent to a non-CS audience. 

The UI explicitly labels every response as grounded or general, because honesty about system limitations was the point of the demo.

`Gemini API` `SentenceTransformers` `FAISS` `pypdf` `Streamlit`

---

### [AI Content Studio — Multi-Agent System](https://github.com/ruthuraraj-ml/Multi-Agent-AI-System-for-Educational-Content-Generation)
![Type](https://img.shields.io/badge/Type-Multi--Agent%20%7C%20LLM%20Systems-darkgreen)

Transforms a topic prompt into structured educational content by orchestrating four specialised agents: Research (RAG) → Image (diffusion diagrams) → Reviewer (structured explanation) → Manager (workflow coordination). 

The project is an experiment in understanding how multi-agent coordination actually works in practice — including where agent handoffs break down, where the RAG pipeline retrieves irrelevant content, and where diffusion-generated diagrams diverge from the intended concept. 

These failure cases are documented alongside the working pipeline.

`CrewAI` `Gemini` `Groq` `FLUX` `RAG` `Streamlit`

---

## 🧠 Generative & Multimodal AI

### [Image Caption Generator — Vision–Language Models](https://github.com/ruthuraraj-ml/Image-Caption-Generation-using-Vision-Language-Models)
![Type](https://img.shields.io/badge/Type-Multimodal%20AI-violet)

Explores automatic caption generation progressing from a CNN–LSTM baseline (InceptionV3 + LSTM) through a Transformer decoder to pretrained vision–language transformers fine-tuned on a small real-world dataset. 

The project deliberately validates learning via controlled overfitting before scaling up — and honestly documents why pretrained models outperform naive fine-tuning on limited data. 

The analysis of what breaks under constrained hardware and small datasets is as central as the results themselves.

`InceptionV3` `LSTM` `Transformer` `Beam Search` `HuggingFace`

---

### [Image Super-Resolution — SRGAN & ESRGAN](https://github.com/ruthuraraj-ml/Image-Super-Resolution-using-SRGAN-and-ESRGAN)
![Type](https://img.shields.io/badge/Type-Generative%20Adversarial%20Networks-violet)

Reconstructs ×4 high-resolution images from low-resolution inputs using SRGAN and ESRGAN (RRDB-based). Built from scratch with patch-based training, warm-up stability phases, and adversarial fine-tuning on the DIV2K dataset. 

The project documents the perceptual trade-offs honestly — results under constrained compute and training time are visibly imperfect, and the analysis explains *why*: mode collapse behaviour, discriminator instability, and the gap between perceptual loss and PSNR metrics. 

Side-by-side and zoomed qualitative comparisons are included without cherry-picking.

`SRGAN` `ESRGAN` `RRDB` `VGG Perceptual Loss` `DIV2K`

---

## 🔬 Research & Development Roadmap

### 🧠 Generative AI & LLM Applications
- **RAG Learning Management System** — Lecture note generation, semantic search, and question creation from educational content *(in development)*
- **VAE for Tabular Data** — Synthetic dataset generation and latent structure learning for structured data
- **Real-ESRGAN Extension** — Domain-specific fine-tuning with perceptual quality metric evaluation

### ⚙️ AI for Mechanical & Manufacturing Engineering
- **Surface Roughness Prediction** — ML/DL models for machining quality from cutting parameters
- **Nano-Additive Bio-Lubricant Modelling** — Tribological performance prediction and eco-lubricant optimisation

> **Research direction:** Building interpretable AI systems that connect mathematical learning principles to real engineering decision-making.

---

## 🛠️ Technology Stack

**Languages:** Python

**ML / DL:**
`scikit-learn` `PyTorch` `TensorFlow/Keras` `NumPy` `Pandas`

**Generative AI & LLMs:**
`Gemini API` `Groq` `HuggingFace` `CrewAI` `SentenceTransformers` `FAISS`

**Generative Models:**
`SRGAN` `ESRGAN` `FLUX` `CNN–LSTM` `Vision–Language Transformers`

**Tools & Platforms:**
`Google Colab` `Streamlit` `Jupyter` `GitHub Actions`

**Domains:**
Regression · Classification · Ensemble Learning · GANs · Multimodal AI · RAG · Agentic AI · Manufacturing AI

---

## 👨‍🏫 About

**R. Ruthuraraj**  
Assistant Professor · Mechanical Engineering · SNS College of Technology  
AICTE QIP Programme — *AI to Generative AI*, IIIT Allahabad

This portfolio documents a self-directed learning journey from classical statistical learning through modern deep learning, generative models, and agentic AI systems — with a long-term focus on applying these methods to engineering problems.

---

## 🙏 Acknowledgements

- AICTE QIP Programme on *AI to Generative AI* — IIIT Allahabad
- SNS College of Technology
- Kaggle · UCI · HuggingFace Datasets
- PyTorch & TensorFlow communities
- SRGAN / ESRGAN research papers

---

⭐ *If you find these projects useful for learning or teaching AI, please star the repositories!*
