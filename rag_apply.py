import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS


key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)

model = genai.GenerativeModel('gemini-2.5-flash-lite')


def load_embedding():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
with st.spinner('Loading embedding model..⌛'):
       embedding_model = load_embedding()
    
st.set_page_config('RAG DEMO',page_icon='🎯', layout='wide' )

st.title('RAG Assistance :blue[Using Embeding and LLM]🤖👾')
st.subheader(':green[Your Intelligent Document Assitance AI]')
uploaded_file = st.file_uploader('Upload the file here in PDF Format' , type=['pdf'])

if uploaded_file:
    pdf = PdfReader(uploaded_file)
    
    raw_text = ''
    for page in pdf.pages:
        raw_text += page.extract_text()
        
    if raw_text.strip():
        doc = Document(page_content=raw_text)
        splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        
       
        splitter = CharacterTextSplitter(chunk_size= 1000, chunk_overlap = 200)
        
        chunk_text = splitter.split_documents([doc])
        
        text = [i.page_content for i in chunk_text]
        vector_db = FAISS.from_texts(text, embedding_model)
        retrive = vector_db.as_retriever()
        st.success('Documents processed and saved successfully!!✅')
        query = st.text_input('Ask me a question ❓')
        
        if query:
            with st.chat_message('human'):
                
                with st.spinner('Analyzling the documnet...🔎'):
                    
                    relevant_data = retrive.invoke(query)
                    content = '\n\n'.join([ i.page_content for i in relevant_data])
                    prompt = f''' 
                    You are an AI expert. Use the generated content {content} to answer the query {query}. If you are sure with the answer, say
                    " I have no content related to this question. Please ask relevant query to answer"
                    
                    Result in bullet points'''
                    
                    response = model.generate_content(prompt)
                    
                    st.markdown('## :green[Results👍]')
                    st.write(response.text)
                    
    else:
        st.warning('Drop the file in PDF format')        

