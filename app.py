import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv  # Tambahkan untuk mengelola environment variables

# Load environment variables dari file .env
load_dotenv()

# Ambil API Key dari environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY tidak ditemukan. Pastikan Anda telah mengatur variabel lingkungan dengan benar.")

# Inisialisasi LLM Groq 
def get_llm(model_name="deepseek-r1-distill-llama-70b", temperature=0.2):
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=temperature
    )

# Global QA Chain
qa_chain = None

# Fungsi proses PDF
def process_pdf(file, model_name="llama-3.3-70b-versatile", temperature=0.2):
    try:
        loader = PyPDFLoader(file.name)
        docs = loader.load_and_split()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        # Gunakan model embeddings yang sebenarnya
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # Ambil 4 dokumen teratas
        
        qa = RetrievalQA.from_chain_type(
            llm=get_llm(model_name=model_name, temperature=temperature),
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )
        return qa, None
    except Exception as e:
        return None, f"Error: {str(e)}"

# Saat user upload file
def upload_file(file, model_name, temperature):
    global qa_chain
    if file is None:
        return "⚠️ Silakan pilih file PDF terlebih dahulu."
    
    qa_chain, error = process_pdf(file, model_name, temperature)
    if error:
        return error
    return f"✅ File berhasil diproses. Model: {model_name}, Temperature: {temperature}. Silakan ajukan pertanyaan."

# Jawab pertanyaan
def ask_question(question):
    if qa_chain is None:
        return "⚠️ Silakan unggah PDF terlebih dahulu."
    if not question.strip():
        return "⚠️ Silakan masukkan pertanyaan."
    
    try:
        result = qa_chain({"query": question})
        answer = result["result"]
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# UI Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:  # Tambahkan tema untuk UI yang lebih baik
    gr.Markdown("## 📄 RAG PDF Q&A - LangChain + Groq API Key")
    
    with gr.Row():
        with gr.Column(scale=3):
            file_input = gr.File(file_types=[".pdf"], label="Unggah File PDF")
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=[
                    "llama-3.3-70b-versatile",
                    "deepseek-r1-distill-llama-70b",
                    "gemma2-9b-it"
                ],
                value="llama-3.3-70b-versatile",
                label="Pilih Model LLM"
            )
            temperature_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                step=0.1,
                value=0.2,
                label="Temperature"
            )
            process_button = gr.Button("Proses PDF", variant="primary")  # Tambahkan variant untuk tampilan yang lebih baik
    
    status = gr.Textbox(label="Status", interactive=False)
    
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Pertanyaan", placeholder="Masukkan pertanyaan Anda tentang dokumen...")
            submit_btn = gr.Button("Tanyakan", variant="primary")
    
    answer = gr.Textbox(label="Jawaban", interactive=False)

    # Update click handler supaya pass model & temperature
    process_button.click(upload_file, inputs=[file_input, model_dropdown, temperature_slider], outputs=status)
    submit_btn.click(ask_question, inputs=question, outputs=answer)
    question.submit(ask_question, inputs=question, outputs=answer)

# Jalankan aplikasi
if __name__ == "__main__":
    demo.launch(share=False)  # Set share=True jika ingin membagikan aplikasi secara publik
