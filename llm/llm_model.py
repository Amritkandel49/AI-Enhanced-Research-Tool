from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import json

load_dotenv()

class ModelSelection:
    def __init__(self, llm_settings:dict):
        self.model_name = llm_settings.get("model", "openai/gpt-oss-20b:free")
        self.temperature = llm_settings.get("temperature", 0.7)
        
        
        if self.model_name == "openai/gpt-oss-20b:free":
            
            self.model = ChatOpenAI(
                base_url= os.getenv('OPENAI_BASE_URL'),
                model = self.model_name,
                temperature=self.temperature,
            )
            
        elif self.model_name == "google/gemini-2.5-flash-lite-preview-06-17":
            self.model = ChatOpenAI(
                base_url="https://lightning.ai/api/v1/",
                api_key=os.getenv("LIGHTNING_API_KEY"),
                model=self.model_name,
                temperature=self.temperature
            )
            
        
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
            
    def get_model(self):
        # print('-'*25)
        # print(self.model)
        return self.model
        
if __name__ == "__main__":
    llm_settings = {
        "model": "FreedomAISVR/Qwable-v1-MXFP4-MOE-GGUF",
        "temperature": 0.7
    }
    model_selection = ModelSelection(llm_settings)
    llm = model_selection.get_model()
    print(llm)
    print(llm.invoke("Hello").content)
    

