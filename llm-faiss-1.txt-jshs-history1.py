from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import asyncio
from dotenv import load_dotenv 

load_dotenv()

loader = TextLoader("files\jshs-history.txt", encoding='utf-8')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =50,
        chunk_overlap  = 0,
        length_function =len
    )
pages = text_splitter.split_documents(documents)

db = FAISS.from_documents(pages , OpenAIEmbeddings())

query = "현재 교장은?"
loop = asyncio.get_event_loop()
docs = loop.run_until_complete( db.asimilarity_search_with_relevance_scores(query) )

for doc, score in docs:
    print(f"{score}\t{doc.page_content}")


from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  

chain = load_qa_chain(llm, verbose=True)


query = "현재 교장은 ? "
docs = db.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query ,res)

query = "초대 교장은 ? "
docs = db.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query, res)

query = "1회 졸업 인원수 ? "
docs = db.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)