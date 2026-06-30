from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    model="openai/gpt-oss-20b:free",
)

print("base_url:", llm.openai_api_base)

print(llm.invoke("Hello").content)