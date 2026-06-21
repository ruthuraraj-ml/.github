<div align="center">

![banner](https://github.com/ruthuraraj-ml/.github/blob/main/profile/banner.svg)

![stats](https://github.com/ruthuraraj-ml/.github/blob/main/profile/stat-strip.svg)

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ruthuraraj/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ruthuraraj-ml)
[![Portfolio](https://img.shields.io/badge/Portfolio-6E40C9?style=for-the-badge&logo=githubpages&logoColor=white)](https://ruthuraraj-ml.github.io/github.io/)
[![LeetCode](https://img.shields.io/badge/LeetCode-FFA116?style=for-the-badge&logo=leetcode&logoColor=black)](https://leetcode.com/u/ceAlpLZW04/)
[![Exercism](https://img.shields.io/badge/Exercism-009CAB?style=for-the-badge&logo=exercism&logoColor=white)](https://exercism.org/profiles/ruthuraraj)


</div>

---

## 💡 Philosophy

> **These projects are not about replicating state-of-the-art benchmarks or shipping polished end products.**

The goal of every repository here is to **build from scratch after genuinely understanding the underlying concepts** — and then to be honest about where things break, what the model cannot do, and why. Limitations are documented as carefully as results, because understanding failure modes is how real learning happens.

This matters especially coming from a Mechanical Engineering background: the instinct here is not to chase accuracy numbers, but to ask *why does this architecture work, where does it fail, and what does that tell us about the problem?* That question drives every project in this portfolio.

---

## ⭐ Dream Project

### [R-B.A.T — RAG-Based Academic Tutor](https://github.com/ruthuraraj-ml/R-BAT-Academic-Tutor)

![Type](https://img.shields.io/badge/Type-RAG%20%7C%20Agentic%20AI%20%7C%20Educational%20AI-blueviolet?style=flat-square)
![LLM](https://img.shields.io/badge/LLM-Gemma3%3A4b%20%7C%20Mistral%20(Ollama)-yellow?style=flat-square)
![Infra](https://img.shields.io/badge/Infra-Fully%20Local%20%7C%20CPU%20Only-success?style=flat-square)
![Repo](https://img.shields.io/badge/Repo-Private%20%7C%20Demo%20on%20Request-red?style=flat-square)

A **fully local, RAG-grounded academic AI system** built for SNS College of Technology — running on CPU with no GPU, no cloud API costs, and no data leaving the institution.

Four purpose-built modes: **Tutor** (RAG Q&A over course PDFs) · **Assessment** (Bloom's Taxonomy-aligned question paper generation with CO mapping) · **Evaluation** (model answer generation grounded in course corpus) · **Presentation** (PPT pipeline with 6 themes × 6 rotating content templates).

**What makes this different:** every mode is a structurally distinct pipeline, not a single chatbot doing everything loosely. Real constraints — no GPU, no API budget, institutional data privacy — are treated as design parameters, not obstacles.

<details>
<summary><b>Architecture & technical details</b></summary>

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│         Midnight Ember UI (Lora + Plus Jakarta Sans)     │
└────────────┬───────────────────────────────┬────────────┘
             │                               │
    ┌────────▼────────┐             ┌────────▼────────┐
    │   RAG Pipeline  │             │  PPT Pipeline   │
    │  (FAISS Index)  │             │  ppt_engine.py  │
    │  Course PDFs    │             │  6 Themes × 6   │
    │  + Syllabus     │             │  Templates      │
    └────────┬────────┘             └────────┬────────┘
             │                               │
    ┌────────▼───────────────────────────────▼────────┐
    │              Local LLM Layer (Ollama)            │
    │   Gemma3:4b — Tutor, Assessment, Evaluation      │
    │   Mistral   — Presentation synthesis             │
    │         CPU-only · No GPU Required               │
    └─────────────────────────────────────────────────┘
```

`Gemma3:4b` `Mistral` `Ollama` `FAISS` `SentenceTransformers` `Streamlit` `ReportLab` `python-pptx`

</details>

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
<!--  ═══════════════════════════════════════════════════════════ -->
<div align="center">
<img src="https://img.shields.io/badge/──────────────────────%20%F0%9F%A7%A0%20%20FOUNDATIONS%20%20%F0%9F%A7%A0%20──────────────────────-1a1a2e?style=for-the-badge&labelColor=1a1a2e&color=534AB7"/>
</div>

*Classical ML · Deep Learning · Representation Learning — the ground every later project is built on.*

---

### 🔹 [XOR Problem — Why Deep Learning Exists](https://github.com/ruthuraraj-ml/XOR-Why-Deep-Learning-Exists)
![Type](https://img.shields.io/badge/Type-Learning%20Theory-5b5fd6?style=flat-square) ![Framework](https://img.shields.io/badge/Framework-scikit--learn-orange?style=flat-square)

A concept-driven demonstration of **why linear models fail and why hidden layers are necessary**. Walks through OR and AND (linearly separable), breaks logistic regression on XOR to show where it fails — then solves it with a single hidden layer MLP. The focus is entirely on decision boundaries and architectural necessity, not accuracy.

`Logistic Regression` `MLP` `Decision Boundaries`

---

### 📐 Classical Machine Learning
![Projects](https://img.shields.io/badge/9%20Projects-Regression%20%7C%20Classification%20%7C%20Deep%20Learning-534AB7?style=flat-square&labelColor=26215C)

<details>
<summary><b>View all 9 projects</b></summary>

| # | Project | Type | Models |
| - | ------- | ---- | ------ |
| 1 | [Advertising Sales Prediction](https://github.com/ruthuraraj-ml/Advertising-Sales-Prediction-using-Linear-Regression) | Regression | Linear Regression |
| 2 | [Bike Sharing Demand Prediction](https://github.com/ruthuraraj-ml/Bike-Sharing-Demand-Prediction) | Time-Pattern Regression | Linear Regression |
| 3 | [Diabetes Prediction](https://github.com/ruthuraraj-ml/Diabetes-Prediction-using-Logistic-Regression) | Medical Classification | Logistic Regression |
| 4 | [Titanic Survival Prediction](https://github.com/ruthuraraj-ml/Titanic-Survival-Prediction-using-Logistic-Regression) | Binary Classification | Logistic Regression |
| 5 | [Wine Quality Prediction](https://github.com/ruthuraraj-ml/Wine-Quality-Prediction-using-Random-Forest-Classifier) | Multiclass Classification | Random Forest |
| 6 | [Health Risk Classification for Insurance Premium Optimization](https://github.com/ruthuraraj-ml/Health-Risk-Classification-for-Insurance-Premium-Optimization) | Medical Risk Classification | LR · DT · RF |
| 7 | [Online Payment Fraud Detection](https://github.com/ruthuraraj-ml/Online-Payment-Fraud-Detection-using-Machine-Learning) | Imbalanced Classification | LR · RF · XGBoost |
| 8 | [NYC Taxi Trip Duration Prediction](https://github.com/ruthuraraj-ml/NYC-Taxi-Trip-Duration-Prediction) | Geospatial Regression | LR · Ridge · Lasso · DT · RF · GB |
| 9 | [Deep Learning for Groundwater Quality Assessment](https://github.com/ruthuraraj-ml/deep-learning-for-groundwater-quality-assessment) | Regression + Multiclass | ANN · BatchNorm · Dropout · Optuna |

</details>

---

### 🧠 Deep Learning & Neural Networks
![Projects](https://img.shields.io/badge/3%20Projects-ANN%20%7C%20RNN%20%7C%20LSTM%20%7C%20BERT-534AB7?style=flat-square&labelColor=26215C)

<details>
<summary><b>View all 3 projects</b></summary>

| # | Project | Focus Area | Techniques |
| - | ------- | ---------- | ---------- |
| 1 | [Twitter Sentiment Analysis](https://github.com/ruthuraraj-ml/Twitter-Sentiment-Analysis-Deep-Learning) | NLP & Sequence Modeling | RNN · LSTM · GRU · BERT · Transfer Learning |
| 2 | [Groundwater Quality Assessment](https://github.com/ruthuraraj-ml/deep-learning-for-groundwater-quality-assessment) | Applied Deep Learning | ANN · Optimizer Comparison · BatchNorm · Dropout · Optuna |
| 3 | [Neural Networks — From Basics to Stabilization](https://github.com/ruthuraraj-ml/Neural-Networks-Demo-PyTorch) | Deep Learning Fundamentals | BatchNorm · Dropout · Optimizers · Training Dynamics |

</details>

---

### 🔤 Embeddings & Representation Learning

### [Word2Vec Embedding Explorer](https://github.com/ruthuraraj-ml/Embedding_Search) · [🌐 Live Demo](https://ruthuraraj-ml.github.io/Embedding_Search/)
![Type](https://img.shields.io/badge/Type-Representation%20Learning-7F77DD?style=flat-square) ![Framework](https://img.shields.io/badge/Framework-PyTorch-red?style=flat-square)

End-to-end Word2Vec (Skip-Gram + Negative Sampling) built from scratch in PyTorch, extended into an **interactive browser-based embedding explorer**. Exports intermediate checkpoints across epochs and visualises how semantic structure gradually emerges from random vectors — nearest neighbours, similarity scoring, analogy solving, geometric clustering.

<details>
<summary><b>Concepts covered</b></summary>

Distributional hypothesis · negative sampling · cosine similarity · semantic clustering · vector arithmetic · geometry of learned representations · effect of training progression on embedding quality

`Word2Vec` `Skip-Gram` `FAISS` `Embedding Visualisation` `WikiText-2`

</details>

---
<!--  ═══════════════════════════════════════════════════════════ -->
<div align="center">
<img src="https://img.shields.io/badge/──────────%20🦙%20%20LLM%20FINE--TUNING%20%26%20PEFT%20%20🦙%20──────────-0d1117?style=for-the-badge&labelColor=0d1117&color=8b5cf6"/>
</div>

*Transformers · LoRA · QLoRA · Quantization · Instruction Tuning*

---
### 🔹 [Parameter-Efficient Fine-Tuning of BERT and Gemma using LoRA & QLoRA](https://github.com/ruthuraraj-ml/Parameter-Efficient-Fine-Tuning-BERT-and-Gemma-using-LoRA-and-QLoRA)

![Type](https://img.shields.io/badge/Type-PEFT%20%7C%20LoRA%20%7C%20QLoRA-8b5cf6?style=flat-square\&labelColor=4c1d95)
![Models](https://img.shields.io/badge/Models-BERT%20Base%20%7C%20Gemma%202B-yellow?style=flat-square)
![Focus](https://img.shields.io/badge/Focus-Instruction%20Tuning%20%7C%20Quantization-success?style=flat-square)

A hands-on exploration of **Parameter-Efficient Fine-Tuning (PEFT)**, progressing from LoRA-based adaptation of **BERT** to QLoRA-based instruction tuning of **Gemma 2B**. The project investigates how large language models can be adapted by training only a tiny fraction of their parameters while significantly reducing memory requirements through 4-bit quantization.

**What makes this different:** the repository documents the complete engineering journey — including an attempted QLoRA implementation on BERT, debugging of bitsandbytes compatibility issues, architectural analysis of encoder vs decoder models, and a successful migration to Gemma 2B. Rather than hiding failed experiments, the project preserves them as learning artifacts.

<details>
<summary><b>Architecture, results & concepts covered</b></summary>

**BERT + LoRA**

* Accuracy: 90.41%
* Trainable Parameters: 591K (0.5372%)

**Gemma 2B + QLoRA**

* Accuracy: 97.04%*
* Trainable Parameters: 6.39M (0.2438%)
* 4-bit NF4 Quantization
* Fine-tuned on a Tesla T4 GPU

**Concepts Covered**

Parameter-Efficient Fine-Tuning · LoRA · QLoRA · 4-bit Quantization · Instruction Tuning · Transformer Architectures · Encoder vs Decoder Models · Hugging Face PEFT · BitsAndBytes · Memory-Efficient LLM Adaptation

`BERT` `Gemma 2B` `LoRA` `QLoRA` `PEFT` `BitsAndBytes` `Transformers` `PyTorch` `Hugging Face`

* Metrics computed on valid generated predictions that could be confidently mapped to sentiment labels.

</details>

---
<!--  ═══════════════════════════════════════════════════════════ -->
<div align="center">
<img src="https://img.shields.io/badge/─────────────%20%F0%9F%A4%96%20%20AGENTIC%20AI%20%26%20LLM%20SYSTEMS%20%20%F0%9F%A4%96%20─────────────-0d1117?style=for-the-badge&labelColor=0d1117&color=1a6e3c"/>
</div>

*Multi-agent orchestration · LangGraph · CrewAI Flows · Reflection loops · Persistent memory*

---

### 🔹 [Enterprise Logistics Orchestration Hub](https://github.com/ruthuraraj-ml/enterprise-logistics-orchestration-hub) ⭐ *Latest*

![Type](https://img.shields.io/badge/Type-CrewAI%20Flows%20%7C%20Decision%20Intelligence-1a6e3c?style=flat-square&labelColor=0a3d1f)
![LLM](https://img.shields.io/badge/LLM-Llama%203.3%20%7C%20Gemma%204%20%7C%20Gemini-yellow?style=flat-square)
![Memory](https://img.shields.io/badge/Memory-SQLite%20Delta%20Reasoning-blueviolet?style=flat-square)

A **CrewAI Flow-powered logistics decision intelligence platform** that transforms supply chain metrics into executive-level optimization playbooks. Parallel analytical branches (inventory + logistics) synchronize through Flow barriers, pass into a memory-aware strategist, and every strategy is reviewed by an independent Critic Agent before a revision cycle triggers when necessary.

**What makes this different:** historical optimization playbooks are actively retrieved and injected into future strategy generation — delta reasoning where recommendations *evolve* from previous decisions rather than restarting from scratch.

<details>
<summary><b>Architecture, multi-LLM design & concepts covered</b></summary>

Multi-LLM cognitive architecture: Llama 3.3 70B for inventory interpretation · Gemma 4 26B for logistics analysis · Gemma 4 31B for strategic synthesis · Gemini Flash Lite for independent validation.

**Concepts covered:** CrewAI Flows · multi-agent orchestration · multi-LLM specialization · reflection-driven strategy revision · persistent SQLite memory · delta reasoning · parallel execution branches · synchronization barriers · executive decision support · logistics optimization · geospatial analytics

`CrewAI` `CrewAI Flows` `Llama 3.3 70B` `Gemma 4` `Gemini Flash Lite` `SQLite` `Gradio` `Pandas` `Pydantic`

</details>

---

### 🔹 [Stateful Market Intelligence Agent](https://github.com/ruthuraraj-ml/stateful-market-intelligence-agent)

![Type](https://img.shields.io/badge/Type-ReAct%20%7C%20LangGraph%20%7C%20RAG-1a6e3c?style=flat-square&labelColor=0a3d1f)
![LLM](https://img.shields.io/badge/LLM-Gemini%20Flash%20Lite-yellow?style=flat-square)
![Memory](https://img.shields.io/badge/Memory-ChromaDB-blueviolet?style=flat-square)

A **LangGraph ReAct competitor intelligence platform** for clothing stores — built in 2 days. Discovers nearby competitors via Apify, enriches with BestTime traffic data (with an inference fallback for missing coverage), runs a reflection-driven validation loop, and stores every completed analysis in ChromaDB as a queryable RAG assistant.

**What makes this different:** the dual-layer traffic strategy (empirical API → inference fallback) guarantees 100% traffic coverage even when primary data is unavailable.

<details>
<summary><b>Concepts covered & stack</b></summary>

LangGraph StateGraph design · ReAct cycle · reflection-driven loop control · dual-layer data resilience · weighted competitive scoring · persistent vector memory · RAG over longitudinal market data · PDF/Excel report generation

`LangGraph` `LangChain` `Gemini` `ChromaDB` `Apify` `BestTime` `Plotly` `Streamlit` `ReportLab` `OpenPyXL`

</details>

---

### 🔹 [Memory-Aware Agentic Travel Planner with Self-Evaluation](https://github.com/ruthuraraj-ml/A-Memory-Aware-Agentic-Travel-Planning-System-with-Self-Evaluation-and-Conditional-Re-Search)

![Type](https://img.shields.io/badge/Type-LangGraph%20%7C%20Reflection%20Loop-1a6e3c?style=flat-square&labelColor=0a3d1f)
![LLMs](https://img.shields.io/badge/LLMs-Gemini%20%7C%20Groq%20LLaMA-yellow?style=flat-square)

A **confidence-scored reflection loop** for travel planning: after initial data collection, the agent evaluates its own information completeness (0–100%), identifies knowledge gaps, and re-searches with targeted queries before generating the final guide. Maximum 2 re-search cycles to prevent runaway API consumption. Every cycle's verdict and gaps are surfaced in an Agent Insights tab — internal reasoning fully auditable.

<details>
<summary><b>Concepts covered & stack</b></summary>

LangGraph StateGraph · confidence-scored self-evaluation · conditional re-search · multi-tool orchestration (weather, search, images) · session memory · deduplicated research merging · transparent AI trace

`LangGraph` `LangChain` `Gemini` `Groq LLaMA` `Tavily` `WeatherAPI` `Pexels` `Streamlit`

</details>

---

### 🔹 [ReAct Web Research Agent](https://github.com/ruthuraraj-ml/ReAct-Web-Research-Agent)

![Type](https://img.shields.io/badge/Type-ReAct%20%7C%20No%20Framework-1a6e3c?style=flat-square&labelColor=0a3d1f)
![LLMs](https://img.shields.io/badge/LLMs-Gemini%20%7C%20Groq%20LLaMA-yellow?style=flat-square)

Autonomous research agent built on the **ReAct paradigm from scratch** — no LangChain, no LangGraph, no framework. Full Thought → Action → Observation → Summary loop per research question using a multi-LLM split: Groq LLaMA 3.3 70B for reasoning steps, Gemini Flash Lite for planning and synthesis.

Honest implementation: limitations (single-pass loop, in-session memory only, no reflection) are documented as the natural next improvements, not hidden.

`Gemini` `Groq LLaMA 3.3 70B` `Tavily` `ReAct Pattern` `Streamlit`

---

### 🔹 [AI Content Studio — Multi-Agent System](https://github.com/ruthuraraj-ml/Multi-Agent-AI-System-for-Educational-Content-Generation)

![Type](https://img.shields.io/badge/Type-Multi--Agent%20%7C%20LLM%20Systems-1a6e3c?style=flat-square&labelColor=0a3d1f)

Four specialised agents — Research (RAG) → Image (diffusion) → Reviewer → Manager — orchestrated to transform a topic prompt into structured educational content. Failure cases (handoff breakdowns, irrelevant RAG retrievals, divergent diffusion diagrams) documented alongside the working pipeline.

`CrewAI` `Gemini` `Groq` `FLUX` `RAG` `Streamlit`

---

### 🔹 [Workshop Assistant — ReAct-lite RAG Agent](https://github.com/ruthuraraj-ml/Workshop-Assistant-RAG-Agent)

![Type](https://img.shields.io/badge/Type-RAG%20%7C%20Agentic%20AI-1a6e3c?style=flat-square&labelColor=0a3d1f) ![Built](https://img.shields.io/badge/Built%20For-Live%20Workshop%20Demo-yellow?style=flat-square)

*First agentic implementation.* Built as a live demo for the final session of a 3-day AI workshop (SNS College of Technology, Jan 2026). Replaces LLM routing with a **FAISS L2 distance threshold** — one fewer API call per query, fully deterministic, and makes the agent's decision logic transparent to a non-CS audience.

`Gemini API` `SentenceTransformers` `FAISS` `pypdf` `Streamlit`

---
<!--  ═══════════════════════════════════════════════════════════ -->
<div align="center">
<img src="https://img.shields.io/badge/──────────%20%F0%9F%A7%AC%20%20GENERATIVE%20%26%20MULTIMODAL%20AI%20%20%F0%9F%A7%AC%20──────────-0d1117?style=for-the-badge&labelColor=0d1117&color=6b21a8"/>
</div>

*RAG pipelines · Multimodal systems · Generative models · Vision-Language*

---

### 🔹 [Paperwise RAG — Multimodal Research Paper Q&A](https://github.com/ruthuraraj-ml/paperwise-rag) *(Repository Temporarily Offline)*

![Type](https://img.shields.io/badge/Type-Multimodal%20RAG-6b21a8?style=flat-square&labelColor=3b0764) ![LLM](https://img.shields.io/badge/LLM-Gemini%20Flash%20Lite-yellow?style=flat-square) ![Embeddings](https://img.shields.io/badge/Embeddings-BGE%20%2B%20FAISS-blue?style=flat-square)

A **multimodal RAG pipeline** handling text, tables, and figures from academic PDFs in a single unified system. Docling preserves structure across all three modalities; BGE embeds into FAISS; Gemini answers with vision-enabled summarisation for figures and tables. Queries each modality separately and merges before generation.

<details>
<summary><b>Engineering details & concepts</b></summary>

Addresses a genuine engineering challenge: most RAG pipelines treat PDFs as plain text, losing tables and figures. OOM crashes on high-resolution page images were fixed by tuning Docling's image scale and disabling full-page rasterisation.

**Concepts:** multimodal document parsing · three-modality retrieval (text / table / figure) · BGE embeddings · FAISS vector search · vision-language generation · memory-efficient PDF processing

`Docling` `BGE Embeddings` `FAISS` `Gemini Flash Lite` `Gradio` `Python`

</details>

---

### 🔹 [Image Super-Resolution — SRGAN & ESRGAN](https://github.com/ruthuraraj-ml/Image-Super-Resolution-using-SRGAN-and-ESRGAN)

![Type](https://img.shields.io/badge/Type-Generative%20Adversarial%20Networks-6b21a8?style=flat-square&labelColor=3b0764)

×4 image reconstruction (SRGAN and RRDB-based ESRGAN) built from scratch with patch-based training, warm-up stability phases, and adversarial fine-tuning on DIV2K. Results under constrained compute are documented honestly — mode collapse, discriminator instability, and the gap between perceptual loss and PSNR are all explained rather than cherry-picked around.

`SRGAN` `ESRGAN` `RRDB` `VGG Perceptual Loss` `DIV2K`

---

### 🔹 [Image Caption Generator — Vision–Language Models](https://github.com/ruthuraraj-ml/Image-Caption-Generation-using-Vision-Language-Models)

![Type](https://img.shields.io/badge/Type-Multimodal%20AI-6b21a8?style=flat-square&labelColor=3b0764)

Caption generation progressing from CNN–LSTM baseline (InceptionV3 + LSTM) through Transformer decoder to pretrained vision–language transformers. Validates learning via controlled overfitting before scaling up. Documents why pretrained models outperform naive fine-tuning on limited data — the analysis of what breaks is as central as the results.

`InceptionV3` `LSTM` `Transformer` `Beam Search` `HuggingFace`

---

## 🛠️ Technology Stack


**Languages**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)

**ML & Data Science**

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat-square&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

**Deep Learning**

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21F?style=flat-square&logo=huggingface&logoColor=black)

**Agentic AI & Orchestration**

![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![CrewAI](https://img.shields.io/badge/CrewAI-6E40C9?style=flat-square&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-2C2C2C?style=flat-square&logoColor=white)

**LLMs & APIs**

![Gemini](https://img.shields.io/badge/Gemini-4285F4?style=flat-square&logo=google&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=flat-square&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square&logoColor=white)

**RAG & Vector DBs**

![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat-square&logoColor=white)

**Frontend & Apps**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-F97316?style=flat-square&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

**Dev Tools**

![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)
![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?style=flat-square&logo=visualstudiocode&logoColor=white)


---

## 🎯 Current Focus

- Applying AI to **engineering, manufacturing, and supply chain optimization**
- Creating **AI workshops and hands-on learning experiences** for students and faculty
- Expanding **multimodal and agentic RAG systems** for research and education
- Evaluating **local and hybrid LLM deployments** (Gemma, Ollama) for cost-efficient agentic systems
- Designing **multi-LLM architectures** that assign specialized models to different reasoning workloads

---

## 🔬 R&D Roadmap

<details>
<summary><b>Agentic AI, Generative AI & Engineering AI roadmap</b></summary>

### 🤖 Agentic AI & Decision Intelligence

* **Enterprise Logistics Hub v2** — Local LLM deployment, advanced telemetry, multi-product optimization, autonomous strategy monitoring
* **Competitor Intelligence Agent v2** — Multi-location comparative analysis, geospatial mapping, longitudinal market memory
* **RAG Learning Management System** — Course-aware note generation, semantic search, question-bank creation, instructor-facing automation
* **Paperwise RAG v2** — Local LLM support (Gemma via Ollama), cross-paper comparison, citation-aware synthesis

### 🧠 Generative AI & LLM Applications

* **VAE for Tabular Data** — Synthetic dataset generation for structured manufacturing datasets
* **Real-ESRGAN Extension** — Domain-specific fine-tuning for engineering imagery

### ⚙️ AI for Mechanical & Manufacturing Engineering

* **Surface Roughness Prediction** — ML/DL models for machining quality from cutting parameters
* **Nano-Additive Bio-Lubricant Modelling** — Tribological performance prediction and eco-lubricant optimization
* **Manufacturing Knowledge Systems** — Agentic assistants and RAG pipelines for engineering education

> **Research direction:** Building interpretable, memory-aware AI systems that bridge modern agentic intelligence with real-world engineering decision-making.

</details>

---

## 👨‍🏫 About

**R. Ruthuraraj** · Assistant Professor · Mechanical Engineering · SNS College of Technology
*AICTE QIP Programme — AI to Generative AI, IIIT Allahabad*

This portfolio documents a **self-directed learning journey** from classical machine learning and statistical modelling to deep learning, generative AI, retrieval-augmented generation (RAG), and modern agentic AI systems.

What began as an effort to learn Python for teaching and engineering applications gradually evolved into a deeper exploration of how intelligent systems reason, retrieve information, collaborate, critique their own outputs, and learn from past decisions. Every project in this portfolio represents not only a completed system, but also the questions, experiments, debugging sessions, architectural redesigns, and lessons learned along the way.

My long-term goal is to bridge **Artificial Intelligence and Engineering**, applying machine learning, generative AI, and agentic systems to manufacturing, supply chain optimization, engineering education, and real-world decision support. This portfolio serves as both a record of that journey and a collection of practical AI systems built through continuous learning and experimentation.

---

## 🙏 Acknowledgements

SNS College of Technology · AICTE QIP Programme · NPTEL Course Instructors · Kaggle · UCI · Hugging Face · CrewAI · LangGraph · LangChain · PyTorch · TensorFlow · Open-Source AI Communities

---

<div align="center">

⭐ *If you find these projects useful for learning, teaching, or exploring AI systems, consider starring the repositories.*

</div>
