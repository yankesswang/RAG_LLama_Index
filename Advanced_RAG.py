from langchain.chains import LLMChain
from langchain_openai import OpenAI,ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema import StrOutputParser
import streamlit as st
import os
load_dotenv()
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_transformers import DoctranTextTranslator
from langchain_core.documents import Document
import os
from llama_index.core import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Document
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import MetadataReplacementPostProcessor,SentenceTransformerRerank
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
import os
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Document
)
from llama_index.core.prompts import LangchainPromptTemplate
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.postprocessor import LongContextReorder
from IPython.display import Markdown, display
from langchain import hub
from langchain.schema import SystemMessage, HumanMessage

llm = ChatOpenAI(temperature=0)
langchain_prompt = hub.pull("rlm/rag-prompt")

def GET_RAG_DATA(file_dir = '/Users/yankesswang/Documents/PYTHON/LLM Project/LangChain/RAG/data'):
    reader = SimpleDirectoryReader(input_dir=file_dir)
    data = reader.load_data()
    docs = []
    for data in reader.iter_data():
        for d in data:
            d.text = d.text.upper()
            docs.append(d)
    return docs

def Prompt_Engineering():
    langchain_prompt = hub.pull("rlm/rag-prompt")
    lc_prompt_tmpl = LangchainPromptTemplate(
            template=langchain_prompt,
            template_var_mappings={"query_str": "question", "context_str": "context"},
    )
    return lc_prompt_tmpl

def NAIVE_RAG_answers(question):
    docs = GET_RAG_DATA()
    node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
    nodes = node_parser.get_nodes_from_documents(documents= docs)
    vector_index= VectorStoreIndex(nodes)
    
    query_engine = vector_index.as_query_engine()
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": Prompt_Engineering()}
    )
    response = query_engine.query(question).response
    return response 



def build_automerging_index(
    docs,
    llm,
    embed_model="local:BAAI/bge-small-en",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(docs)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=6,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    reorder = LongContextReorder()
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[reorder]
    )
    return auto_merging_engine


def ADVANCED_RAG_answers(question):
    docs = GET_RAG_DATA()
    index = build_automerging_index(
        docs,
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
        save_dir="./merging_index",
    )

    auto_merging_query_engine = get_automerging_query_engine(index, similarity_top_k=6)
    query_engine_tools = [
        QueryEngineTool(
            query_engine=auto_merging_query_engine,
            metadata=ToolMetadata(
                name="llm_expert",
                description="Research Paper Essay",
            )
        )
    ]
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        use_async=True,
    )
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": Prompt_Engineering()}
    )
    response = query_engine.query(question)
    return response

def output_engineering(input):
    chat = ChatOpenAI(
        model= 'gpt-3.5-turbo',
        temperature= 0,
    )
    system_prompt = """
        You are a professional writer tasked with transforming an existing answer into a well-structured, easy-to-understand response. Your goal is to ensure that the response covers all the main points in a reader-friendly manner without losing important details or insights. 

        To achieve this, you will separate the different parts of the answer using numbered sections, with each number corresponding to a specific question or point being addressed. Follow this structure:
        - Use clear and concise language to explain each point.
        - Incorporate examples or illustrations where appropriate to aid understanding.
        - Ensure a logical flow and coherence throughout the response.
        - Format the text appropriately (e.g., using bullet points, line breaks, etc.) to improve readability.

        Here is an example of the desired format:

        1. How to improve language models:
        - Through fine-tuning on specific domains or tasks
        - By incorporating retrieval-augmented generation (RAG) techniques

        2. What is the structure of language models:
        - They often use transformer architectures
        - Consisting of an encoder and decoder with attention mechanisms
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=str(input))
    ]
    response = chat.invoke(messages)
    output = response.content
    return output

def main():
    st.set_page_config(layout="wide")
    st.title("LLM Paper answer machine")
    question = st.text_area("Ask a question about LLM Paper:", height=200)
    if question:

        naive_answer = NAIVE_RAG_answers(question)
        advanced_answer = ADVANCED_RAG_answers(output_engineering(question))  
        col1, col2 = st.columns(2)


        with col1:
            with st.expander("See Naive RAG Answer", expanded=True):
                st.markdown("#### Naive RAG Answer")
                st.info(naive_answer)

  
        with col2:
            with st.expander("See Advanced RAG Answer", expanded=True):
                st.markdown("#### Advanced RAG Answer")
                st.success(advanced_answer)
       
if __name__ == "__main__":
    main()




