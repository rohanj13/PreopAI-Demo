import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
from tqdm.auto import tqdm
import tiktoken
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from time import sleep

# Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pdf_folder_path = "references/"

# Embedding model
embed_model = "text-embedding-3-small"

# System prompt (prepopulated)
default_system_prompt = """You are the anesthesiologist seeing this patient in the preoperative clinic 2 weeks before the date of operation. The patients have already taken their routine preoperative investigations and the findings are listed within the clinical summary.
Your role is to evaluate the clinical summary and give the preoperative anesthesia instructions for the following patient targeted to your fellow medical colleagues. You are to follow strictly the guidelines.
Your instructions should consist of the following components:
1. Provide a traffic light status for the surgery. If there is a risk and the patient needs to be seen by a Doctor or a Nurse, its red, if further tests are required, its yellow; if the patient is healthy and ready for surgery, its green.
2. Fasting instructions - list instructions based on the number of hours before the time of the listed surgery
3. Suitability for preoperative carbohydrate loading — yes/no.
4. Medication instructions — name each medication and give the instructions for the day of the operation and days leading up to the operation as required.
5. Any instructions for the healthcare team—for example, preoperative blood group matching, arranging for preoperative dialysis, or standby post-operative high dependency/ICU beds.
6. Provide the RCRI, ASA, DASI and STOP-BANG scores for the patient. If you cannot calculate it, provide the extra information you need to calculate it.
Your instructions are the final instructions, explain the reasoning for your 
if you are uncertain, explain what further information you require.
If the medical condition is already optimized, there is no need to offer further optimization. If there are no relevant instructions in any of the above categories, leave it blank and write NA"""

# Streamlit UI
st.title("PreopAI Demo")

query = st.text_area("Enter Clinical Query", height=500)
system_prompt = st.text_area("System Prompt", value=default_system_prompt, height=600)
run_button = st.button("Run AI Analysis")

if run_button and query.strip():
    with st.spinner("Processing..."):

        # Load and process PDFs (cache for speed)
        @st.cache_resource
        def load_chunks():
            loader = PyPDFDirectoryLoader(pdf_folder_path)
            dataset = loader.load()
            data = []
            for doc in dataset:
                data.append({
                    'reference': doc.metadata['source'].replace('rtdocs/', 'https://'),
                    'text': doc.page_content
                })
            tokenizer = tiktoken.get_encoding('cl100k_base')
            def tiktoken_len(text):
                tokens = tokenizer.encode(text, disallowed_special=())
                return len(tokens)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=tiktoken_len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = []
            for idx, record in enumerate(data):
                texts = text_splitter.split_text(record['text'])
                chunks.extend([{
                    'id': str(uuid4()),
                    'text': texts[i],
                    'chunk': i,
                    'reference': record['reference']
                } for i in range(len(texts))])
            return chunks

        chunks = load_chunks()

        # Pinecone setup
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = "preopai-index-py"
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                vector_type="dense",
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        index = pc.Index(index_name)

        # OpenAI client
        client = OpenAI(api_key=openai_api_key)

        # Embed query
        res = client.embeddings.create(input=[query], model=embed_model)
        xq = res.data[0].embedding

        # Retrieve from Pinecone
        res = index.query(
            namespace="__default__",
            vector=xq,
            top_k=10,
            include_metadata=True
        )
        contexts = [item['metadata']['text'] for item in res['matches']]
        augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query

        # Run LLM
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query},
            ]
        )

        st.subheader("AI Output")
        st.text_area("Response", value=response.choices[0].message.content, height=500, width=700)