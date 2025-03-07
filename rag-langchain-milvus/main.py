from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
import os
from dotenv import load_dotenv

load_dotenv()


def get_embeddings_model():
    return OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))


def get_llm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=os.environ.get("OPENAI_API_KEY"))


def read_document(file_path):
    loader = TextLoader(file_path)
    return loader.load()


def process_chunks():
    raw_documents = read_document("./data/quantum_computing.txt")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(raw_documents)


def get_db():
    return "./database/raptai.db"


def create_embeddings_and_upsert():
    documents = process_chunks()
    embeddings_model = get_embeddings_model()

    # Configure Milvus connection
    return Milvus.from_documents(
        documents,
        embeddings_model,
        connection_args={"uri": get_db()},
        collection_name="quantum_computing_docs",  # Name your collection
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 100}  # Adjust nlist based on dataset size
        }
    )


def user_query():
    return 'Please explain Difference between classical computer and quantum computer?'


def create_retriever():
    vector_store = create_embeddings_and_upsert()
    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})


def retrieve_relevant_docs():
    retriever = create_retriever()
    query = user_query()
    docs = retriever.invoke(query)
    return docs


def prepare_prompt():
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context: {context} Question: {question} """
    )
    return prompt


@chain
def qa(input):
    docs = retrieve_relevant_docs()
    prompt = prepare_prompt()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    formatted = prompt.invoke({"context": docs, "question": input})
    answer = llm.invoke(formatted)
    return answer


# run it
result = qa.invoke(user_query())
print(result.content)
