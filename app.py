# Streamlit LLM Chat with Document App

import streamlit as st
import openai
import requests
import os
from typing import List, Dict
from uuid import uuid4
import io

# For PDF and DOCX support
try:
	import PyPDF2
except ImportError:
	PyPDF2 = None
try:
	import docx
except ImportError:
	docx = None

# --- UI Styling ---
st.set_page_config(page_title="Chat with Your Document", page_icon="📄", layout="wide")
st.markdown("""
	<style>
	.stChatMessage {background: #f5f7fa; border-radius: 10px; margin-bottom: 10px; padding: 10px;}
	.stButton>button {background: linear-gradient(90deg,#2563eb,#1e40af); color: white; border-radius: 6px; font-weight: 600;}
	.stTextInput>div>input {border-radius: 6px; font-size: 1.1em;}
	.stFileUploader {border-radius: 6px;}
	.stForm {background: #fff; border-radius: 10px; box-shadow: 0 2px 8px #e0e7ef; padding: 20px;}
	.stMarkdown {font-size: 1.05em;}
	</style>
""", unsafe_allow_html=True)

# --- Accessibility: ARIA labels and instructions ---
st.markdown("""
<div aria-label='Document Chat Application' role='main'>
<p><b>Instructions:</b> Upload a document (.txt, .pdf, .docx, .csv, .tsv) and ask questions about its content. Your data is processed securely and locally.</p>
</div>
""", unsafe_allow_html=True)

# --- Helper Functions ---

# --- Document Loaders ---
"""
Document loader functions for various file types.
All functions return extracted text for LLM context.
"""

@st.cache_data(show_spinner=False)
def load_txt(file) -> str:
	"""Load text from a .txt file."""
	try:
		return file.read().decode("utf-8")
	except Exception:
		return file.read().decode("latin-1")

@st.cache_data(show_spinner=False)
def load_csv(file, delimiter=',') -> str:
	"""Extract CSV/TSV as text."""
	import pandas as pd
	try:
		df = pd.read_csv(file, delimiter=delimiter)
		return df.to_csv(index=False)
	except Exception as e:
		return f"[Error loading CSV/TSV: {e}]"

@st.cache_data(show_spinner=False)
def load_tsv(file) -> str:
	"""Extract TSV as text."""
	return load_csv(file, delimiter='\t')

@st.cache_data(show_spinner=False)
def load_pdf(file) -> str:
	"""Extract text from a PDF file."""
	if not PyPDF2:
		raise ImportError("PyPDF2 is required for PDF support. Please install it with 'pip install PyPDF2'.")
	try:
		reader = PyPDF2.PdfReader(file)
		text = "\n".join(page.extract_text() or "" for page in reader.pages)
		return text
	except Exception as e:
		return f"[Error loading PDF: {e}]"

@st.cache_data(show_spinner=False)
def load_docx(file) -> str:
	"""Extract text from a DOCX file."""
	if not docx:
		raise ImportError("python-docx is required for DOCX support. Please install it with 'pip install python-docx'.")
	try:
		doc = docx.Document(file)
		text = "\n".join([para.text for para in doc.paragraphs])
		return text
	except Exception as e:
		return f"[Error loading DOCX: {e}]"

def load_document(file, filetype: str) -> str:
	"""Dispatch loader based on file type."""
	if filetype == "txt":
		return load_txt(file)
	elif filetype == "pdf":
		return load_pdf(file)
	elif filetype == "docx":
		return load_docx(file)
	elif filetype == "csv":
		return load_csv(file)
	elif filetype == "tsv":
		return load_tsv(file)
	else:
		raise ValueError("Unsupported file type. Supported: txt, pdf, docx, csv, tsv.")


# --- Ollama Client ---
@st.cache_resource(show_spinner=False)
def get_ollama_client():
	# No API key needed for local Ollama
	return True

def get_ollama_response(prompt: str, model: str = "phi3:mini") -> str:
	url = "http://localhost:11434/api/generate"
	payload = {"model": model, "prompt": prompt, "stream": False}
	try:
		response = requests.post(url, json=payload)
		response.raise_for_status()
		return response.json().get("response", "")
	except Exception as e:
		return f"[Ollama error: {e}]"

def split_text(text: str, max_tokens: int = 1500) -> List[str]:
	# Simple splitter for context window
	paragraphs = text.split('\n\n')
	chunks, chunk = [], ""
	for para in paragraphs:
		if len(chunk) + len(para) < max_tokens:
			chunk += para + "\n\n"
		else:
			chunks.append(chunk)
			chunk = para + "\n\n"
	if chunk:
		chunks.append(chunk)
	return chunks

def find_relevant_chunks(chunks: List[str], query: str, top_k: int = 2) -> List[str]:
	# Naive keyword search for demo; replace with embeddings for production
	scored = [(chunk, chunk.lower().count(query.lower())) for chunk in chunks]
	scored.sort(key=lambda x: x[1], reverse=True)
	return [c for c, s in scored[:top_k] if s > 0] or [chunks[0]]

def build_prompt(context_chunks: List[str], chat_history: List[Dict], user_query: str) -> str:
	context = "\n---\n".join(context_chunks)
	history = "\n".join([f"User: {m['user']}\nAI: {m['ai']}" for m in chat_history])
	prompt = f"You are a helpful assistant. Use the following document context to answer.\n\nDocument:\n{context}\n\n{history}\nUser: {user_query}\nAI:"
	return prompt

def get_openai_response(prompt: str, api_key: str) -> str:
	client = get_openai_client(api_key)
	response = client.completions.create(
		model="gpt-3.5-turbo-instruct",
		prompt=prompt,
		max_tokens=256,
		temperature=0.2,
		stop=["User:", "AI:"]
	)
	return response.choices[0].text.strip()

# --- Session State Management ---
if 'session_id' not in st.session_state:
	st.session_state.session_id = str(uuid4())
if 'chat_history' not in st.session_state:
	st.session_state.chat_history = []
if 'doc_chunks' not in st.session_state:
	st.session_state.doc_chunks = []
if 'doc_name' not in st.session_state:
	st.session_state.doc_name = None
if 'saved_contexts' not in st.session_state:
	st.session_state.saved_contexts = {}


# --- Sidebar: Model Selection, API Key, and Context Management ---
st.sidebar.title("🧠 LLM Model & Context")
model_choice = st.sidebar.selectbox("Choose LLM Model", ["OpenAI", "Ollama (phi3:mini)"])
api_key = ""
if model_choice == "OpenAI":
	api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if st.sidebar.button("Save Current Context"):
	if st.session_state.doc_name:
		st.session_state.saved_contexts[st.session_state.doc_name] = {
			'chat_history': st.session_state.chat_history.copy(),
			'doc_chunks': st.session_state.doc_chunks.copy()
		}
		st.sidebar.success(f"Context for '{st.session_state.doc_name}' saved.")
if st.session_state.saved_contexts:
	selected = st.sidebar.selectbox("Restore Context", ["-"] + list(st.session_state.saved_contexts.keys()))
	if selected and selected != "-":
		ctx = st.session_state.saved_contexts[selected]
		st.session_state.chat_history = ctx['chat_history'].copy()
		st.session_state.doc_chunks = ctx['doc_chunks'].copy()
		st.session_state.doc_name = selected
		st.sidebar.success(f"Restored context for '{selected}'.")

# --- Main UI ---
st.title("📄 Chat with Your Document")
st.write("Upload a document and chat with its content using OpenAI's LLM. Supported formats: TXT, PDF, DOCX, CSV, TSV.")



uploaded_file = st.file_uploader(
	"Upload document",
	type=["txt", "pdf", "docx", "csv", "tsv"],
	accept_multiple_files=False,
	key="file_uploader",
	help="Supported: TXT, PDF, DOCX, CSV, TSV"
)

if uploaded_file:
       filetype = uploaded_file.name.split(".")[-1].lower()
       try:
	       # For PDF/DOCX/CSV/TSV, need to reset file pointer after reading
	       if filetype in ["pdf", "docx", "csv", "tsv"]:
		       file_bytes = uploaded_file.read()
		       file_obj = io.BytesIO(file_bytes)
		       doc_text = load_document(file_obj, filetype)
	       else:
		       doc_text = load_document(uploaded_file, filetype)
	       st.session_state.doc_chunks = split_text(doc_text)
	       st.session_state.doc_name = uploaded_file.name
	       # Load or initialize chat history for this document
	       if uploaded_file.name in st.session_state.saved_contexts:
		       st.session_state.chat_history = st.session_state.saved_contexts[uploaded_file.name]['chat_history'].copy()
	       else:
		       st.session_state.chat_history = []
	       # Save context immediately after file load
	       st.session_state.saved_contexts[uploaded_file.name] = {
		       'chat_history': st.session_state.chat_history.copy(),
		       'doc_chunks': st.session_state.doc_chunks.copy()
	       }
	       st.success(f"Loaded document: {uploaded_file.name}")
       except Exception as e:
	       st.error(f"Failed to load document: {e}")


if model_choice == "OpenAI":
	if not api_key:
		st.warning("Please enter your OpenAI API key in the sidebar.")
		st.stop()
if not st.session_state.doc_chunks:
	st.info("Please upload a .txt document to begin.")
	st.stop()


# --- Chat Interface ---
if st.session_state.doc_name:
	st.subheader(f"Chatting with: {st.session_state.doc_name}")
else:
	st.subheader("No document loaded yet.")

col1, col2 = st.columns([4, 1])
with col2:
	if st.button("🧹 Clear Conversation History"):
		st.session_state.chat_history = []
		# Also update saved context for this document
		if st.session_state.doc_name:
			st.session_state.saved_contexts[st.session_state.doc_name]['chat_history'] = []
		st.experimental_rerun()

with col1:
	for msg in st.session_state.chat_history:
		st.markdown(f"<div class='stChatMessage'><b>User:</b> {msg['user']}</div>", unsafe_allow_html=True)
		st.markdown(f"<div class='stChatMessage' style='background:#e0e7ef;'><b>AI:</b> {msg['ai']}</div>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
		user_query = st.text_input(
			"Your question about the document:",
			key="user_query",
			help="Type your question and press Enter or click Send.",
			placeholder="e.g. Summarize the document, find keywords, etc."
		)
		submitted = st.form_submit_button("Send")

def get_openai_response_fast(prompt: str, api_key: str) -> str:
	# Use ChatCompletion if available for faster response
	try:
		client = get_openai_client(api_key)
		response = client.chat.completions.create(
			model="gpt-3.5-turbo",
			messages=[{"role": "system", "content": "You are a helpful assistant."},
					  {"role": "user", "content": prompt}],
			max_tokens=256,
			temperature=0.2,
			stream=False
		)
		return response.choices[0].message.content.strip()
	except Exception:
		# fallback to Completion
		return get_openai_response(prompt, api_key)

if submitted and user_query:
	with st.spinner("Thinking..."):
		relevant_chunks = find_relevant_chunks(st.session_state.doc_chunks, user_query)
		prompt = build_prompt(relevant_chunks, st.session_state.chat_history, user_query)
		if model_choice == "OpenAI":
			try:
				answer = get_openai_response_fast(prompt, api_key)
			except Exception as e:
				st.error(f"OpenAI API error: {e}")
				answer = "[Error: Could not get response from OpenAI API.]"
		else:
			answer = get_ollama_response(prompt, model="phi3:mini")
		st.session_state.chat_history.append({"user": user_query, "ai": answer})
		# Save updated conversation to context
		if st.session_state.doc_name:
			st.session_state.saved_contexts[st.session_state.doc_name]['chat_history'] = st.session_state.chat_history.copy()
		st.markdown(f"<div class='stChatMessage'><b>User:</b> {user_query}</div>", unsafe_allow_html=True)
		st.markdown(f"<div class='stChatMessage' style='background:#e0e7ef;'><b>AI:</b> {answer}</div>", unsafe_allow_html=True)
