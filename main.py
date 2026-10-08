import chromadb
import os
import llama_index
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex
)

from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import QueryBundle
from llama_index.core.llms import MockLLM #--> to prevent open api error since QueryFusionRetriever asks for an external llm to build multiple queries outta single one
print("All imports done ")

# initializing directories if not exist
DATA_DIR = "./storage/data/"
CHROMA_DIR = "./storage/vectorDB/"
DOCSTORE_PATH = "./storage/docstore.json" #--> to keep track of files
os.makedirs(DATA_DIR,exist_ok=True)
os.makedirs(CHROMA_DIR,exist_ok=True)


# setting default model for embedding creation
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5", device="cuda"
)

# defining chunk size and overlap size
Settings.node_parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
)

# creates a dummy LLM 
Settings.llm = MockLLM()

# initializing chromadb client
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR) #--> creates a chromadb client
chroma_collection = chroma_client.get_or_create_collection(name="retriever") #--> creates a chroma collection

# connecting chromadb to llama_index 

vectorstore = ChromaVectorStore(chroma_collection) #instantiates a LlamaIndex-compatible vector store object

# Load existing docstore for file tracking, or initialize a new one
if os.path.exists(DOCSTORE_PATH):
    docstore = SimpleDocumentStore.from_persist_path(DOCSTORE_PATH)
    print("Loaded existing docstore tracking cache.")
else:
    docstore = SimpleDocumentStore()
    print("Initialized new docstore tracking cache.")

# Ingestion pipeline handles tracking, chunking, and embedding 
pipeline = IngestionPipeline(
    transformations=[
        Settings.node_parser,
        Settings.embed_model,
    ],
    vector_store=vectorstore,
    docstore=docstore,
    docstore_strategy=DocstoreStrategy.UPSERTS,
) #--> ingestion pipeline does not require storagecontext which is used for inserting data into vectorDB
# 1. Pipeline automatically checks hashes:
#    - Does nothing if files already exist in docstore
#    - Embeds only new / modified files into ChromaDB

# pdf reader using pdf parser 
pdf_reader = SimpleDirectoryReader(
    input_dir=DATA_DIR,file_extractor={".pdf":PyMuPDFReader()},required_exts=".pdf"
)

documents = pdf_reader.load_data() #--> contains all the extracted pdf data 

pipeline.docstore.persist(DOCSTORE_PATH)

# making multiple chunks from single document to manage easily 
nodes = pipeline.run(documents=documents, show_progress=True)
print(f"this is the number of total chunks created {len(nodes)}")

index = VectorStoreIndex.from_vector_store(vectorstore) #connects LlamaIndex to data that is already embedded and saved in vector database without re-chunking or re-embedding


# creating hybrid retriever 
dense_retriever = index.as_retriever(similarity_top_k=10) # --> gives a dense load of top matching semantic chunks
all_nodes = list(docstore.docs.values())
keyword_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=10) #--> checks for every keyword and matches them

hybrid_retriever = QueryFusionRetriever(
    retrievers=[dense_retriever,keyword_retriever],
    similarity_top_k=10,
    num_queries=1,  # 1 query = gives only single query to retrievers
    mode="reciprocal_rerank",  # Fuses using Reciprocal Rank Fusion and get highest matching top k outputs
    use_async=False,
)

# defining a re-ranker to get the top most query 
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3, device="cuda"
)# gives top three chunks that matches our query the most 

query_str = "What does Natural language processing (NLP) systems pull out"

# 1. Retrieve candidates across vector + BM25 stores (top 10)
initial_nodes = hybrid_retriever.retrieve(query_str)

# 2. Rerank candidate chunks with the cross-encoder down to top_n (top 3)
final_nodes = reranker.postprocess_nodes(
    nodes=initial_nodes,
    query_bundle=QueryBundle(query_str)
)

# 3. Access chunks, scores, and metadata
for idx, node_with_score in enumerate(final_nodes, 1):
    print(f"--- Result {idx} (Score: {node_with_score.score:.4f}) ---")
    print(f"File: {node_with_score.node.metadata.get('file_name', 'Unknown')}")
    print(f"Page: {node_with_score.node.metadata.get('page_label', 'N/A')}")
    print(f"Content:\n{node_with_score.node.text}\n")