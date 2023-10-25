from dotenv import load_dotenv 

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import asyncio
load_dotenv()

loader = TextLoader("files\jshs-history.txt", encoding='utf-8')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 50,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = False,
    )
docs = text_splitter.split_documents(documents)


# Chroma에 문서 로드
db = Chroma.from_documents(docs, OpenAIEmbeddings())

query = "초대 교장은?"
loop = asyncio.get_event_loop()
docs = loop.run_until_complete( db.asimilarity_search_with_relevance_scores(query) )
for doc, score in docs:
    print(f"Document: {doc}\tRelevance Score: {score}")


from langchain.chat_models import ChatOpenAI

llm_model= ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  

query = "초대 교장은?"
docs = db.similarity_search(query)
chain = load_qa_chain(llm_model , chain_type="stuff")
res = chain.run(input_documents=docs, question=query)
print( query,res)
