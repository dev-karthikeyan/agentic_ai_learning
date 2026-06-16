import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

load_dotenv()

api_key=os.getenv("GOOGLE_API_KEY")

MODEL = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def get_weather(city : str) -> str :
    """Get the weather in a city"""
    return f"get weather in the {city}"
    
agent = create_agent(

    model = MODEL,
    tools = [get_weather]

)    

response = agent.invoke({"messages" : [{"role" : "user" , "content" : "hey there"}]})

print(response)

