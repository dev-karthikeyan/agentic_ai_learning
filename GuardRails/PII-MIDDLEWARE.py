from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents.middleware import PIIMiddleware
from langchain.agents import create_agent
from langchain_core.tools import tool 

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-120b")


@tool
def customer_lookup(query : str ) -> str :
    """lookup for the customer information"""
    return f"customer record found for query {query}"


agent = create_agent(

     model = model ,
     tools = [customer_lookup] ,
     middleware = [
        PIIMiddleware(
          
             "email" ,
             strategy = "redact" ,
             apply_to_input = True , 
    ),
        PIIMiddleware(

             "credit_card" , 
             strategy = "mask" ,
             apply_to_input = True , 

    ),
        PIIMiddleware(
            "api_key" ,
            detector=r"sk-[a-zA-Z0-9]{32}" ,
            strategy = "block" ,
            apply_to_input = True ,

    )
]
)

result = agent.invoke ({
    
    "messages" : [{
        "role" : "user" ,
        "content" : "hey karthiaathi@gmail.com and 4111 1111 1111 1111 this is my account number"
    }]
    })

print("------printing------")
print(result["messages"][-1].content)

for msg in result["messages"]:
    print(msg)
