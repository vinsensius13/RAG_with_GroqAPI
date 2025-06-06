<h1 align="center"> Retrieval-Augmented Generation with Gradio and Groq API Key</h1>
<p align="center"> Natural Language Processing Project</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">

</div>

### Name : Arifian Saputra
### Tech Stack : Python, Gradio, LangChain, HuggingFace Embedding, FAISS vector store

---

### 1. Analysis about how the project works
- This project implements a Retrieval-Augmented Generation (RAG) system using LangChain and Groq API.
- The user uploads a PDF document through a Gradio UI.
- The document is split into text chunks using CharacterTextSplitter.
- The text chunks are embedded using HuggingFaceEmbeddings and stored in a FAISS vectorstore.
- A retriever fetches the most relevant text chunks when the user asks a question.
- A selected LLM (via Groq API) generates the answer based on the retrieved chunks and the user query.
- The project supports dynamic selection of LLM model and temperature from the UI.
- This allows users to experiment with different model behaviors and response creativity.

### 2. Analysis about how different every model works on Retrieval-Augmented Generation

```python
def get_llm(model_name="deepseek-r1-distill-llama-70b", temperature=0.2):
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=temperature
    )
```
- Model used : ```[llama-3.3-70b-versatile, deepseek-r1-distill-llama-70b, gemma2-9b-it]```

2.1 Analysis on ```llama-3.3-70b-versatile``` : 
- Generates highly detailed and structured answers.
- Very good at understanding context from the document.
- Slightly slower response but very reliable for complex queries.
- Best for deep and nuanced questions.

2.2 Analysis on ```deepseek-r1-distill-llama-70b``` : 
- Faster than llama-3.3.
- Generates concise and to-the-point answers.
- Sometimes less detailed, but still good factual accuracy.
- Suitable for short, direct Q&A tasks.

2.3 Analysis on ```gemma2-9b-it``` : 
- Provides short and formal answers.
- Tends to be more deterministic, less creative.
- Good for documents with clear factual content.
- Less suitable for generating explanations or deep reasoning.

### 3. Analysis about how temperature works

```python
def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2 # Change the temperature value here and analzye
    )
```

3.1 Analysis on higher temperature 
- Generates more creative and varied answers.
- Increased chance of adding "hallucinations" or information not strictly in the document.
- Useful for brainstorming or generating diverse outputs.
- Less suitable for tasks requiring strict factual accuracy.

3.2 Analysis on lower temperature
- Generates consistent and factual answers.
- More deterministic: similar input produces similar output.
- Best for applications that require accuracy and reproducibility.
- Less creative, responses tend to be more "boring" but safe.

### 4. How to run the project

- Clone this repository with : 

```git
git clone https://github.com/vinsensius13/RAG_with_GroqAPI.git
```

- Copy the ```.env.example``` file and rename it to ```.env```

```
GROQ_API_KEY=your-groq-api-key
```

- Fill the ```GROQ_API_KEY``` with your Groq API Key, find it here : https://console.groq.com/keys

- Install dependencies:
    pip install -r requirements.txt


- Run the application:
    python app.py

---


- Open the application in your browser:
        http://127.0.0.1:7860/

---
        
- Upload a PDF document.
- Select the LLM model and temperature using the UI.
- Click "Proses PDF" to load the document.
- Type your question and click "Tanyakan" to get an answer. 

- This project was tested with:
    - LangChain Community version X.X.X
    - FAISS-cpu version X.X.X
    - sentence-transformers version X.X.X
    - Gradio version X.X.X
- The project supports dynamic model and temperature selection via the Gradio UI.
- For educational purposes.