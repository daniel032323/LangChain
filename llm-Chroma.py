from dotenv import load_dotenv 

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
load_dotenv()

loader = TextLoader("jshs-history.txt", encoding='utf-8')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 50,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = False,
    )
docs = text_splitter.split_documents(documents)

# 임베딩 함수 생성
embedding_function = OpenAIEmbeddings( )

# Chroma에 문서 로드
db = Chroma.from_documents(docs, embedding_function)


query = "초대 교장은?"
# docs = db.similarity_search(query)
loop = asyncio.get_event_loop()
docs = loop.run_until_complete( db.asimilarity_search_with_relevance_scores(query) )
for doc, score in docs:
    print(f"Document: {doc}\tRelevance Score: {score}")

retriever= db.as_retriever()

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  
 # Modify model_name if you have access to GPT-4


chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = retriever,
    return_source_documents=True)    

ret=chain.run("초대 교장은?")

print(ret)
                
