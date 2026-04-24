from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from rich import print
load_dotenv()

file_paths = ["research_paper_1.pdf", "research_paper_2.pdf"]
final_docs = []

for path in file_paths:
    loader = PyPDFLoader(path)
    final_docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
)

data = text_splitter.split_documents(final_docs)

embeding_model = MistralAIEmbeddings()

chromadb = Chroma(
    persist_directory="research_rag",
    embedding_function= embeding_model,
    collection_name="chromadb"
)

chromadb.add_documents(data)

retriever = chromadb.as_retriever(
    search_type = "mmr",
    search_kwargs = {'k': 3}
)


