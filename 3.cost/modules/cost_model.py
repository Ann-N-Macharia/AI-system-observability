# Prices as of June 2026. Always verify at platform.openai.com/docs/pricing.
MODEL_PRICING = {
    "gpt-4o":      {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output":  0.60 / 1_000_000},
}
USD_TO_KSH = 130.0

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model not in MODEL_PRICING:
        raise ValueError("Model untracked.")
    pricing     = MODEL_PRICING[model]
    input_cost  = prompt_tokens  * pricing["input"]
    output_cost = completion_tokens * pricing["output"]
    return round(input_cost + output_cost, 8)
