from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class ModelSelection:
    def __init__(self, llm_settingd:dict):
        self.model_name = llm_settingd.get("model", "openai/gpt-oss-20b:free")
        self.temperature = llm_settingd.get("temperature", 0.7)
    





# model = ChatOpenAI(model="gpt-oss-20b")
model = ChatOpenAI(
  base_url= os.getenv('OPENAI_BASE_URL'),
  model = "openai/gpt-oss-20b:free",
  temperature=0.7,
)

result = model.invoke("What is the capital of France? Explain in 5 short sentences.")

print(result.content)