from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import PromptTemplate, Settings
from llama_index.llms.openai import OpenAI
import os
import streamlit as st
import random

# ---------------------------------------
# SETUP AND CONFIGURATION
# ---------------------------------------

# install llama-index : pip install llama-index
# install streamlit : pip install streamlit
# if you want to use OpenAI for your LLMs and embedding models, get an OpenAI API key (not free) : https://platform.openai.com/api-keys
# and put it into an OPENAI_API_KEY environment variable:
# - "export OPENAI_API_KEY=XXXXX" on linux, "set OPENAI_API_KEY=XXXXX" on Windows
# - in a conda env: 'conda env config vars set OPENAI_API_KEY=api_key', then 'conda deactivate', then 'conda activate {env_name}'
# run script with : streamlit run app.py

PRODUCT = "PRODUCT"
PRODUCT_DESC = "PRODUCT_DESC"
DATA_DIR = "./workitems"
INDEX_DIR = "./storage"
LLM_MODEL_NAME = "gpt-4o-mini"

llm = OpenAI(model = LLM_MODEL_NAME)
Settings.llm = llm

# to also change the embedding model:

#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#embedding_name = "OrdalieTech/Solon-embeddings-base-0.1"
#embed_model = HuggingFaceEmbedding(model_name=embedding_name)
#Settings.embed_model = embed_model

@st.cache_data
def load_index():
    """
    Load or create an index from documents in the specified directory.

    If the index directory does not exist, it reads documents from the data directory,
    creates a new index, and persists it. If the index directory exists, it loads the
    index from storage.

    """
    if not os.path.exists(INDEX_DIR):
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=INDEX_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context)
    return index

index = load_index()

def prepare_template():
    """
    Prepare a prompt template for the QA system.
    """
    text_qa_template_str = """
    Tu es GuruDoc, le ProjectOwner du produit {PRODUCT} {PRODUCT_DESC}.
    Tu as l'ensemble des connaissances en français.
    Tu aides les intervenants à faire des recherches sur la base de connaissances et de ressources disponibles.
    L’un d’eux t’a posé cette question : {query_str}
    Voilà tout ce que tu sais à ce sujet :
    --------
    {context_str}
    --------
    À partir de ces connaissances à toi, et uniquement à partir d’elles, réponds en français à la question.
    Écris une réponse claire et concise.
    """
    qa_template = PromptTemplate(text_qa_template_str)
    return qa_template


st.markdown("""
            <div style='text-align: center;'><h1>GuruDoc</h1></div>
            """
            , unsafe_allow_html=True)

# Initialize session state messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Oui ?"}]

# Capture user input and append it to session state messages
if prompt := st.chat_input("Que veux-tu savoir ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

avatar_filepath = "media/gourou.png"
# Display chat messages with appropriate avatars
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatar_filepath if message["role"] == "assistant" else '💰'):
        st.write(message["content"])


qa_template = prepare_template()
query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=2)

if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant", avatar=avatar_filepath):
        with st.spinner("Vous avez osé sortir GuruDoc de son sommeil ! Patientez deux secondes le temps qu’il se réveille"):
            response = query_engine.query(prompt)
        if response:
            # get source files used to generate the answer, and link to the corresponding youtube videos:
            source_files = [node.metadata['file_name'] for node in response.source_nodes]
            source_files = list(set(source_files))
            text_to_add = "\n\nTu pourras peut-être trouver plus d’infos dans ces documents:"
            for i, file in enumerate(source_files):
                text_to_add += f"\n<a target='_blank'>{file}</a>"
                if i < len(source_files) - 1:
                    text_to_add += " ou"
            st.markdown(response.response + text_to_add, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response.response})

            # to display content used to generate the answer:
            #for node in response.source_nodes:
            #    print("\n----------------")
            #    print(f"Texte utilisé pour répondre : {node.text}")


    
