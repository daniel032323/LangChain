from dotenv import load_dotenv 

from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import asyncio


load_dotenv()

loader = TextLoader("unsu.txt", encoding='utf-8')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap  = 30,
        length_function =len
    )
docs = text_splitter.split_documents(documents)
db = FAISS.from_documents(docs, OpenAIEmbeddings())
query = "아내가 좋아하는 음식?"
loop = asyncio.get_event_loop()
docs = loop.run_until_complete( db.asimilarity_search_with_relevance_scores(query) )
for doc, score in docs:
     print(f"Document: {doc}\tRelevance Score: {score}")



from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

llm = ChatOpenAI(model_name="gpt-4", temperature=2)  

chain = load_qa_chain(llm, chain_type="stuff")


query = "아내가 먹고 싶어 하는 음식?"
docs = db.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( res)

query = "주인공의 직업은?"
docs = db.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( res)

query = "지은이?"
docs = db.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( res)