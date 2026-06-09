# 🚀 ruthuraraj-ml — Machine Learning & Generative AI Portfolio

> *A structured learning journey: **Classical ML → Deep Learning → Representation Learning → Generative AI → Agentic AI Systems***

**R. Ruthuraraj** · Assistant Professor, Mechanical Engineering · SNS College of Technology  
*QIP Programme on 'AI to Generative AI' — IIIT Allahabad*

---

## 🌐 Portfolio Website

**[ruthuraraj-ml.github.io/github.io](https://ruthuraraj-ml.github.io/github.io/)**

A dedicated portfolio website showcasing projects, workshops, certifications, and ongoing AI research activities beyond what is covered in this GitHub profile.

Highlights include:

* **3 AI workshops designed and delivered during 2026** at SNS College of Technology (January, March, and May)
* **R-B.A.T (RAG-Based Academic Tutor)** featuring four operational modes: Tutor, Assessment, Presentation, and Evaluation, powered by local LLMs
* **Professional Development & Certifications:** AICTE QIP (*AI to Generative AI*, IIIT Allahabad), NPTEL Gold (Top 2%), and NPTEL Silver ×3
* Project reports, architecture diagrams, technical blogs, and learning notes
* Links to active profiles on **[GitHub](https://github.com/ruthuraraj-ml) · [LeetCode](https://leetcode.com/u/ceAlpLZW04/) · [Exercism](https://exercism.org/profiles/ruthuraraj)**

The website serves as a central hub documenting my ongoing journey in Machine Learning, Generative AI, Agentic Systems, and Engineering Applications of AI.

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

### [Enterprise Logistics Orchestration Hub](https://github.com/ruthuraraj-ml/enterprise-logistics-orchestration-hub) ⭐ *Latest*
![Type](https://img.shields.io/badge/Type-CrewAI%20Flows%20%7C%20Parallel%20Branches%20%7C%20Decision%20Intelligence-darkgreen) ![LLM](https://img.shields.io/badge/LLM-Llama%203.3%20%7C%20Gemma%204%20%7C%20Gemini%20Flash%20Lite-yellow) ![Memory](https://img.shields.io/badge/Memory-SQLite%20(Delta%20Reasoning)-blueviolet)

A **CrewAI Flow-powered logistics decision intelligence platform** built using a real-world supply chain dataset to transform operational metrics into executive-level optimization playbooks. The system combines inventory analytics, delivery risk analysis, geospatial route intelligence, reflection-driven validation, and persistent organizational memory to generate actionable logistics and inventory strategies.

The architecture orchestrates specialized agents across parallel analytical branches. Inventory and logistics insights are generated independently, synchronized through Flow barriers, and passed into a memory-aware strategist that synthesizes optimization recommendations using both current analytics and historical strategies stored in SQLite. Every strategy is reviewed by an independent Critic Agent, which identifies risks, unsupported assumptions, and missing considerations before triggering a revision cycle when necessary. 

A multi-LLM cognitive architecture assigns different models to specialized reasoning workloads: Llama 3.3 70B for inventory interpretation, Gemma 4 26B for logistics analysis, Gemma 4 31B for strategic synthesis, and Gemini 3.1 Flash Lite for independent validation. The platform also includes portfolio-level product comparison, historical strategy exploration, and live telemetry that exposes the complete agent orchestration process in real time.

**What makes this different:** instead of treating memory as passive storage, historical optimization playbooks are actively retrieved and injected into future strategy generation, enabling delta reasoning where recommendations evolve from previous decisions rather than restarting from scratch.

**Concepts covered:** CrewAI Flows · Multi-Agent orchestration · multi-LLM specialization · reflection-driven strategy revision · persistent SQLite memory · delta reasoning · parallel execution branches · synchronization barriers · executive decision support · logistics optimization · geospatial analytics · historical strategy comparison

`CrewAI` `CrewAI Flows` `Llama 3.3 70B` `Gemma 4` `Gemini Flash Lite` `SQLite` `Gradio` `Pandas` `Pydantic` `Supply Chain Analytics`

---

### [Competitor Intelligence Agent](https://github.com/ruthuraraj-ml/competitor-intelligence-agent) 
![Type](https://img.shields.io/badge/Type-ReAct%20%7C%20LangGraph%20%7C%20RAG-darkgreen) ![LLM](https://img.shields.io/badge/LLM-Gemini%20Flash%20Lite-yellow) ![Memory](https://img.shields.io/badge/Memory-ChromaDB-blueviolet)

A **LangGraph-powered ReAct-style agentic system** for retail market analysis — built in 2 days as a fully functional competitor intelligence platform for clothing stores. The system discovers nearby competitors via Apify's Google Places crawler, enriches each profile with empirical traffic data from BestTime API (with an analytical inference fallback engine for missing coverage), and runs a **reflection-driven validation loop** before generating AI executive reports.

The architecture separates five concerns into distinct LangGraph nodes — Search, Enrichment, Analytics, Reflection, and Summary — with conditional routing that loops back to Search if the Reflection Engine returns `INSUFFICIENT`. Every completed analysis is vectorised into ChromaDB and becomes queryable through a Gemini RAG assistant across sessions.

**What makes this different:** the dual-layer traffic strategy (empirical API → inference fallback) guarantees 100% traffic coverage even when primary data is unavailable — a real engineering decision, not an assumption that data will always be there.

**Concepts covered:** LangGraph StateGraph design · ReAct cycle (Reason → Act → Observe → Reflect) · reflection-driven loop control · dual-layer data resilience · weighted competitive scoring · persistent vector memory · RAG over longitudinal market data · PDF/Excel report generation

`LangGraph` `LangChain` `Gemini` `ChromaDB` `Apify` `BestTime` `Plotly` `Streamlit` `ReportLab` `OpenPyXL`

---

### [Reflective Travel Assistant](https://github.com/ruthuraraj-ml/reflective-travel-assistant)
![Type](https://img.shields.io/badge/Type-ReAct%20%7C%20LangGraph%20%7C%20Multi--Tool-darkgreen) ![LLMs](https://img.shields.io/badge/LLMs-Gemini%20%7C%20Groq%20LLaMA-yellow)

A **memory-aware agentic travel planning system** built on LangGraph — combining real-time weather data, web search, LLM reasoning, self-evaluation, and conditional re-search to generate personalized, visually enriched travel itineraries.

The core feature is a **confidence-scored reflection loop**: after initial data collection, the agent evaluates its own information completeness (0–100% confidence score), identifies specific knowledge gaps (e.g., transportation options, seasonal events), and re-searches with targeted queries before generating the final guide. The loop runs for at most 2 cycles to prevent runaway API consumption, and every cycle's verdict, confidence score, and gaps are surfaced in the Agent Insights tab — making the agent's internal reasoning fully auditable.

**Concepts covered:** LangGraph StateGraph · confidence-scored self-evaluation · conditional re-search · multi-tool orchestration (weather, search, images) · session memory · deduplicated research merging · transparent AI trace

`LangGraph` `LangChain` `Gemini` `Groq LLaMA` `Tavily` `WeatherAPI` `Pexels` `Streamlit`

---

### [ReAct Web Research Agent](https://github.com/ruthuraraj-ml/ReAct-Web-Research-Agent)
![Type](https://img.shields.io/badge/Type-ReAct%20%7C%20Agentic%20AI-darkgreen) ![LLMs](https://img.shields.io/badge/LLMs-Gemini%20%7C%20Groq%20LLaMA-yellow)

An autonomous research agent built on the **ReAct (Reasoning + Acting)** paradigm from scratch — without using LangChain, LangGraph, or any agent framework. Given a topic, the agent generates research questions, then for each question runs a full Thought → Action → Observation → Summary loop using a **multi-LLM architecture**: Groq LLaMA 3.3 70B for fast reasoning steps, Gemini Flash Lite for planning and synthesis, and Tavily for live web retrieval.

The project is an honest implementation of the ReAct loop as it actually works — including where it doesn't: the current loop runs once per question without re-querying on poor results, memory is in-session only, and there is no reflection step. These are documented as the natural next improvements, not hidden.

**Concepts covered:** ReAct agent design · multi-LLM role specialisation · tool use and web grounding · structured research memory · trace generation · report synthesis

`Gemini` `Groq LLaMA 3.3 70B` `Tavily` `ReAct Pattern` `Streamlit`

---

### [Workshop Assistant — ReAct-lite RAG Agent](https://github.com/ruthuraraj-ml/Workshop-Assistant-RAG-Agent)
![Type](https://img.shields.io/badge/Type-RAG%20%7C%20Agentic%20AI-darkgreen) ![Built](https://img.shields.io/badge/Built%20For-Live%20Workshop%20Demo-yellow)

Built as a **live demo for the final session** of a 3-day AI workshop (*From Machine Learning to AI Agents*, SNS College of Technology, Jan 2026). The agent answers questions grounded in workshop PDFs and notebooks, using a heuristic ReAct-style decision loop.

The core design decision — replacing an LLM routing call with a **FAISS L2 distance threshold** — was deliberate: it eliminates one API call per query, is fully deterministic, and makes the agent's reasoning transparent to a non-CS audience.

`Gemini API` `SentenceTransformers` `FAISS` `pypdf` `Streamlit`

---

### [AI Content Studio — Multi-Agent System](https://github.com/ruthuraraj-ml/Multi-Agent-AI-System-for-Educational-Content-Generation)
![Type](https://img.shields.io/badge/Type-Multi--Agent%20%7C%20LLM%20Systems-darkgreen)

Transforms a topic prompt into structured educational content by orchestrating four specialised agents: Research (RAG) → Image (diffusion diagrams) → Reviewer (structured explanation) → Manager (workflow coordination).

The project is an experiment in understanding how multi-agent coordination actually works in practice — including where agent handoffs break down, where the RAG pipeline retrieves irrelevant content, and where diffusion-generated diagrams diverge from the intended concept. These failure cases are documented alongside the working pipeline.

`CrewAI` `Gemini` `Groq` `FLUX` `RAG` `Streamlit`

---

## 🧠 Generative & Multimodal AI

### [Paperwise RAG — Multimodal Research Paper Q&A](https://github.com/ruthuraraj-ml/paperwise-rag) 
![Type](https://img.shields.io/badge/Type-Multimodal%20RAG%20%7C%20LLM%20Systems-violet) ![LLM](https://img.shields.io/badge/LLM-Gemini%20Flash%20Lite-yellow) ![Embeddings](https://img.shields.io/badge/Embeddings-BGE%20%2B%20FAISS-blue)

A **multimodal RAG pipeline for research paper question answering** — handling text, tables, and figures from academic PDFs in a single unified system. Papers are parsed with Docling (preserving structure across all three modalities), embedded with BGE embeddings into a FAISS index, and answered via Gemini with vision-enabled summarisation for figures and tables.

The system addresses a genuine engineering challenge: most RAG pipelines treat PDFs as plain text, losing the structured information in tables and the visual content in figures. Paperwise RAG queries each modality separately and merges results before generation, so answers can draw on the full informational content of a paper — not just its prose.

Built to handle large academic PDFs robustly: OOM crashes on high-resolution page images were fixed by tuning Docling's image scale and disabling full-page rasterisation.

**Concepts covered:** multimodal document parsing · three-modality retrieval (text / table / figure) · BGE embeddings · FAISS vector search · vision-language generation · RAG pipeline engineering · memory-efficient PDF processing

`Docling` `BGE Embeddings` `FAISS` `Gemini Flash Lite` `Gradio` `Python`

---

### [Image Caption Generator — Vision–Language Models](https://github.com/ruthuraraj-ml/Image-Caption-Generation-using-Vision-Language-Models)
![Type](https://img.shields.io/badge/Type-Multimodal%20AI-violet)

Explores automatic caption generation progressing from a CNN–LSTM baseline (InceptionV3 + LSTM) through a Transformer decoder to pretrained vision–language transformers fine-tuned on a small real-world dataset.

The project deliberately validates learning via controlled overfitting before scaling up — and honestly documents why pretrained models outperform naive fine-tuning on limited data. The analysis of what breaks under constrained hardware and small datasets is as central as the results themselves.

`InceptionV3` `LSTM` `Transformer` `Beam Search` `HuggingFace`

---

### [Image Super-Resolution — SRGAN & ESRGAN](https://github.com/ruthuraraj-ml/Image-Super-Resolution-using-SRGAN-and-ESRGAN)
![Type](https://img.shields.io/badge/Type-Generative%20Adversarial%20Networks-violet)

Reconstructs ×4 high-resolution images from low-resolution inputs using SRGAN and ESRGAN (RRDB-based). Built from scratch with patch-based training, warm-up stability phases, and adversarial fine-tuning on the DIV2K dataset.

The project documents the perceptual trade-offs honestly — results under constrained compute and training time are visibly imperfect, and the analysis explains *why*: mode collapse behaviour, discriminator instability, and the gap between perceptual loss and PSNR metrics. Side-by-side and zoomed qualitative comparisons are included without cherry-picking.

`SRGAN` `ESRGAN` `RRDB` `VGG Perceptual Loss` `DIV2K`

---

## 🎯 Current Focus

- **Applying AI**, machine learning, and GenAI methods to engineering, manufacturing, and supply chain optimization
- **Creating AI workshops**, demonstrations, and **hands-on learning experiences** for students and faculty
- Expanding **multimodal and agentic RAG systems** for research, education, and knowledge management
- Evaluating **local and hybrid LLM deployments** (Gemma, Ollama, and open-weight models) for cost-efficient agentic systems
- Building **multi-agent AI systems** using CrewAI Flows, reflection loops, and persistent memory
- Designing **multi-LLM architectures** that assign specialized models to different reasoning workloads
- Developing **memory-aware decision intelligence platforms** for logistics, business, and engineering applications
- Exploring agent orchestration patterns including routing, synchronization barriers, and **adaptive workflow execution**

---

## 🔬 Research & Development Roadmap

### 🤖 Agentic AI & Decision Intelligence

* **Enterprise Logistics Orchestration Hub v2** — Local LLM deployment, advanced telemetry dashboard, multi-product optimization campaigns, and autonomous strategy monitoring
* **Competitor Intelligence Agent v2** — Multi-location comparative analysis, geospatial mapping, specialist multi-agent architecture, and longitudinal market memory
* **RAG Learning Management System** — Course-aware note generation, semantic search, question-bank creation, and instructor-facing content automation
* **Paperwise RAG v2** — Local LLM support (Gemma via Ollama), cross-paper comparison, citation-aware synthesis, and enhanced Gradio experience

### 🧠 Generative AI & LLM Applications

* **Multi-Agent Decision Intelligence Systems** — Reflection loops, persistent memory, adaptive workflows, and model-specialized reasoning
* **VAE for Tabular Data** — Synthetic dataset generation and latent structure learning for structured manufacturing datasets
* **Real-ESRGAN Extension** — Domain-specific fine-tuning and perceptual quality evaluation for engineering imagery

### ⚙️ AI for Mechanical & Manufacturing Engineering

* **Surface Roughness Prediction** — ML/DL models for machining quality from cutting parameters
* **Nano-Additive Bio-Lubricant Modelling** — Tribological performance prediction and eco-lubricant optimization
* **Manufacturing Knowledge Systems** — Agentic assistants and RAG pipelines for engineering education and industrial decision support

> **Research direction:** Building interpretable, memory-aware AI systems that bridge modern agentic intelligence with real-world engineering decision-making.

---

## 🛠️ Technology Stack

**Languages:**
`Python`

**Machine Learning & Deep Learning:**
`Scikit-Learn` `PyTorch` `TensorFlow/Keras` `NumPy` `Pandas`

**Agentic AI & Orchestration:**
`CrewAI` `CrewAI Flows` `LangGraph` `LangChain` `ReAct` `Reflection Loops` `Persistent Memory` `Multi-Agent Orchestration`

**Generative AI & LLMs:**
`Gemini API` `Groq` `HuggingFace` `Ollama` `SentenceTransformers` `BGE Embeddings`

**RAG & Knowledge Systems:**
`FAISS` `ChromaDB` `Docling` `Hybrid Retrieval` `Multimodal RAG`

**Generative Models:**
`VAE` `SRGAN` `ESRGAN` `FLUX` `CNN-LSTM` `Vision-Language Transformers`

**Data & Storage:**
`SQLite` `FAISS` `ChromaDB`

**Frontend & Applications:**
`Gradio` `Streamlit` `Jupyter` `Google Colab`

**Document & Reporting:**
`ReportLab` `OpenPyXL`

**Dev Tools:**
`Git` `GitHub` `GitHub Actions`

**Domains:**
Agentic AI · Decision Intelligence · RAG Systems · Multi-Agent Systems · Manufacturing AI · Supply Chain Analytics · Generative AI · Educational AI

---

## 👨‍🏫 About

**R. Ruthuraraj**
Assistant Professor · Mechanical Engineering · SNS College of Technology
AICTE QIP Programme — *AI to Generative AI*, IIIT Allahabad

This portfolio documents a **self-directed learning journey** from classical machine learning and statistical modelling to deep learning, generative AI, retrieval-augmented generation (RAG), and modern agentic AI systems.

What began as an effort to learn Python for teaching and engineering applications gradually evolved into a deeper exploration of how intelligent systems reason, retrieve information, collaborate, critique their own outputs, and learn from past decisions. Every project in this portfolio represents not only a completed system, but also the questions, experiments, debugging sessions, architectural redesigns, and lessons learned along the way.

My long-term goal is to bridge **Artificial Intelligence and Engineering**, applying machine learning, generative AI, and agentic systems to manufacturing, supply chain optimization, engineering education, and real-world decision support. This portfolio serves as both a record of that journey and a collection of practical AI systems built through continuous learning and experimentation.

---

## 🙏 Acknowledgements

* SNS College of Technology
* AICTE QIP Programme on *AI to Generative AI* — IIIT Allahabad
* My NPTEL Course Instructors, whose teaching sparked my journey into Python, Machine Learning, and AI
* Kaggle · UCI · Hugging Face Datasets
* CrewAI, LangGraph, LangChain, and Open-Source AI Communities
* PyTorch & TensorFlow Communities
* Research communities advancing Generative AI, RAG, and Agentic Systems

---

⭐ *If you find these projects useful for learning, teaching, or exploring AI systems, consider starring the repositories and sharing feedback.*
