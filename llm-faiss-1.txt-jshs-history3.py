from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from dotenv import load_dotenv 

load_dotenv()

embeddings = HuggingFaceEmbeddings()
fDB = FAISS.load_local("faiss-jshs", embeddings)
DB = VectorStoreIndexWrapper(vectorstore=fDB)
llm_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  

response=DB.query("현재 교장은 ?", llm=llm_model)
print(response)

response=DB.query("초대 교장은 ?", llm=llm_model)
print(response)

