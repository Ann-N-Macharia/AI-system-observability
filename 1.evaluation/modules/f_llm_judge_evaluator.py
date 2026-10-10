# =========================
# IMPORT LIBRARIES
# =========================

#Python modules:
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

#Local Modules:
from modules.v_llm_judge_prompt import JUDGE_SYSTEM_PROMPT

# =============================
# CREATE JUDGE FUNCTION
# ============================
def llm_judge(question: str, reference: str, hypothesis: str,
              judge_model: str) -> dict:
    judge_llm = ChatOpenAI(model=judge_model, temperature=0, max_tokens=400)
    user_prompt = f"""
            USER SYMPTOM QUERY: {question}
            REFERENCE PROTOCOL:   {reference}
            AI ASSISTANT RESPONSE: {hypothesis}
            """
    response = judge_llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])
    raw = response.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(m.group()) if m else {
            "correctness": 0, "groundedness": 0, "relevance": 0,
            "helpfulness": 0, "overall": 0, "reasoning": "JSON Parse failure"
        }

# ===========
# TESTING
# ===========
if __name__ == "__main__":
    from dotenv import load_dotenv
    print(JUDGE_SYSTEM_PROMPT)
    load_dotenv()

    # Runnable demo. Requires OPENAI_API_KEY in your environment or a loaded .env file.
    question  = "A patient reports a dry cough lasting two weeks and a low-grade fever. What should they do?"
    reference = ("Persistent cough beyond 14 days with fever warrants a mandatory referral "
                 "to respiratory clinical testing.")
    hypothesis = ("Since the cough has lasted two weeks and is accompanied by fever, the patient "
                  "should be referred for respiratory clinical testing.")
    model = "gpt-4o-mini"
    scores = llm_judge(question, reference, hypothesis,model)
    print(scores)