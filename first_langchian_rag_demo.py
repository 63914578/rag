# Document Loading-->Splitting-->Storage-->Retrieval-->Output
from langchain.document_loaders import PyPDFLoader
 
# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("C:/Users/YANGX224/LangChain/普通感冒01.pdf"),
    PyPDFLoader("C:/Users/YANGX224/LangChain/普通感冒02.pdf"),
    PyPDFLoader("C:/Users/YANGX224/LangChain/普通感冒03.pdf"),
    PyPDFLoader("C:/Users/YANGX224/LangChain/普通感冒04.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
     
# Define the Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
 
#Create a split of the document using the text splitter
splits = text_splitter.split_documents(docs)
 
from langchain.vectorstores import Chroma
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
 
embed = QianfanEmbeddingsEndpoint(
    qianfan_ak='QMMpHFQebv2bmM7vgILhMFPa',
    qianfan_sk='eJGa26RDw7MKn2TsLWiVW3d84wpiE3vp'
)
 
persist_directory = 'C:/Users/YANGX224/LangChain/docs/chroma/'
 
# Create the vector store
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embed,
    persist_directory=persist_directory
)
 
print(vectordb._collection.count())
 
question = "什么是普通感冒？"
 
docs = vectordb.similarity_search(question,k=3)
 
# Check the length of the document
len(docs)
 
# Check the content of the first document
docs[0].page_content
 
# Persist the database to use it later
vectordb.persist()
 
smalldb = Chroma.from_documents(docs, embedding=embed)
question = "普通感冒如何治疗？"
smalldb.max_marginal_relevance_search(question,k=2, fetch_k=4)
