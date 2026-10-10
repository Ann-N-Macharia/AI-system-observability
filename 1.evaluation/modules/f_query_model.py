# =========================
# IMPORT LIBRARIES
# =========================

#Python Modules:
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

#local modules:
from modules.v_evaluation_prompt import SYSTEM_PROMPT


# =============================
# CREATE MODEL QUERY FUNCTION
# ============================
def query_model(model_name: str, question: str) -> str:
    llm = ChatOpenAI(model=model_name, temperature=0, max_tokens=300)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
    response = llm.invoke(messages)
    return response.content

# ===========
# TESTING
# ===========
if __name__=="__main__":
    from dotenv import load_dotenv
    print(SYSTEM_PROMPT)
    load_dotenv()

    response = query_model("gpt-4o", "What symptoms suggest immediate medical attention for a person with suspected malaria?")
    print(response)
