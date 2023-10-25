from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv 
load_dotenv()

loader = TextLoader("files\jshs-history.txt", encoding='utf-8')

DB = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding= HuggingFaceEmbeddings()
        ).from_loaders([loader])

# 파일로 저장
DB.vectorstore.save_local("faiss-jshs")

llm_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  

response=DB.query("현재 교장은?", llm=llm_model)
print(response)
response=DB.query("초대 교장은?", llm=llm_model)
print(response)
quit()
