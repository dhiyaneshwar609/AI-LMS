# LeeLearn – AI-Powered Learning Management System

**An intelligent Learning Management System with a Retrieval-Augmented Generation (RAG) based AI Tutor.**

LeeLearn is an AI-powered Learning Management System designed to improve the teaching and learning experience. It provides a platform where teachers can manage learning materials and students can access educational resources, generate assessments and interact with an AI-powered tutor.

## 🚀 Key Features

### 1. AI Tutor
- Interactive AI-powered chatbot for student doubts.
- RAG-based question answering using uploaded learning materials.
- Context-aware responses based on retrieved course content.
- Conversation-based learning experience.
- Local LLM inference using Ollama.

### 2. RAG-Based Knowledge Retrieval
- PDF document processing and text extraction.
- OCR support for scanned documents.
- Intelligent document chunking.
- Semantic embeddings using Sentence Transformers.
- Vector-based retrieval.
- FAISS-based indexing and similarity search.
- Source-aware answer generation.

### 3. Automatic Quiz Generation
- AI-generated multiple-choice questions.
- Automatic answer generation.
- Question generation from learning materials.
- Quiz-oriented learning support.

### 4. Assignment Generation
- Automated assignment generation.
- Short-answer and long-answer question generation.
- Course-content-based question creation.

### 5. Learning Management
- Teacher and student interfaces.
- Course and learning material management.
- PDF upload and download.
- Student chat history and folder management.
- Student performance monitoring.

### 6. Additional Features
- YouTube learning resource recommendations.
- Learning assistance through interactive conversations.
- Automated educational content generation.

## 🏗️ System Architecture

```text
             Teacher / Student
                     |
                     v
              LeeLearn Web UI
                     |
                     v
                Flask Backend
                     |
          +----------+----------+
          |          |          |
          v          v          v
       AI Tutor   Quiz       Assignment
          |      Generator    Generator
          |
          v
       RAG Pipeline
          |
          v
     Semantic Retrieval
          |
          v
     FAISS Vector Index
          |
          v
   Sentence Transformer
          |
          v
    Retrieved Context
          |
          v
       Local LLM
        (Ollama)
          |
          v
     Generated Answer
```

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| LLM Runtime | Ollama |
| LLM Models | Gemma 3 4B, Qwen 3 4B |
| RAG Framework | Custom RAG Pipeline |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| PDF Processing | PyMuPDF |
| OCR | Tesseract |
| Document Generation | Python-docx, python-pptx, ReportLab |

## ⚙️ Installation and Setup

### Prerequisites

- Python
- Git
- Ollama
- A supported local LLM model

### 1. Clone the Repository

```bash
git clone https://github.com/dhiyaneshwar609/AI-LMS.git
```

```bash
cd AI-LMS
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install and Configure Ollama

Install Ollama from:

https://ollama.com/

Download a supported model:

```bash
ollama pull gemma3:4b
```

Alternatively:

```bash
ollama pull qwen3:4b
```

Ensure the Ollama service is running.

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
AI_LMS_SECRET_KEY=your_generated_secret_key
```

Generate a secure secret key rather than using the example text.

### 6. Run the Application

```bash
python app.py
```

Open the application in your browser at the local address displayed by Flask.

## 📂 Project Structure

```text
AI-LMS/
│
├── app.py
├── database.py
├── ai_tutor.py
├── ai_worker.py
│
├── hosted_llm.py
├── learning_mode.py
│
├── generator_short.py
├── generator_long.py
├── multi_question.py
│
├── rag_v2_processor.py
├── rag_v2_build_embeddings.py
├── rag_v2_search.py
│
├── rag_data_v2/
│
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│
├── tests/
│
├── requirements.txt
└── README.md
```

## 🧠 RAG Implementation

LeeLearn uses Retrieval-Augmented Generation to improve the relevance of AI Tutor responses.

The workflow consists of:

1. Extracting text from educational PDF documents.
2. Processing and splitting documents into chunks.
3. Generating semantic embeddings.
4. Building a vector index.
5. Retrieving relevant content for student queries.
6. Passing retrieved knowledge to the local language model.
7. Generating context-grounded responses.

**Embedding Model:** `all-MiniLM-L6-v2`

**Embedding Dimension:** 384

**LLM Runtime:** Ollama

## 🔒 Privacy and Security

- API credentials are configured through environment variables.
- Flask session security uses a configurable secret key.
- SQLite is used for application data storage.
- Local LLM inference is supported through Ollama.

Do not commit `.env` files, student records, passwords or private credentials.

## 🎯 Project Objective

The primary objective of LeeLearn is to integrate Artificial Intelligence into the traditional Learning Management System by providing intelligent tutoring, automated educational content generation and personalized learning assistance.

## 👨‍💻 Development

LeeLearn is an evolving academic AI project. Features and improvements are continuously being developed.

## 📜 License

Choose an appropriate open-source license before distributing this project. Ensure that all included educational materials are authorized for redistribution.

## 📸 LeeLearn UI Previews

### AI Tutor Interface
![LeeLearn AI Tutor](screenshots/ai-tutor.png)

### RAG-Based AI Tutor
![LeeLearn RAG AI Tutor](screenshots/rag-ai-tutor.png)
