import os
from dotenv import load_dotenv
import streamlit as st
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, load_tools
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.schema import HumanMessage

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')

llm = ChatOpenAI(temperature=0.6, model_name='gpt-3.5-turbo')
tool_names = ['serpapi', 'llm-math']
tools = load_tools(tool_names, llm)
agent = initialize_agent(tools, llm, agent_type='zero-shot-react-description', verbose=False)

loader = PyPDFLoader('UP-Facts.pdf')
document = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(document)
embeddings = OpenAIEmbeddings()
store = FAISS.from_documents(docs, embeddings)

def is_real_time_query(query):
    real_time_keywords = ["today", "current", "live", "now", "price", "weather", "news"]
    return any(keyword in query.lower() for keyword in real_time_keywords)

st.title("Query Response System")
user_query = st.text_input("Ask me whatever you want:")

if st.button("Submit"):
    if is_real_time_query(user_query):
        response = agent.invoke({"input": user_query})
        output_text = ''.join(response['output'])
        st.write(f"Real-Time Response: {output_text}")
    else:
        similar_docs = store.similarity_search(user_query, k=4)
        combined_input = f"User query: {user_query}\n\nRelevant document chunks:\n" + \
                         "\n".join([doc.page_content for doc in similar_docs])
        response = llm([HumanMessage(content=combined_input)])
        st.write(f"PDF Response: {response}")
