from langchain_ollama import OllamaLLM


def analyze_chunks_for_mutation(chunks):
    """
    Send the combined context of the retrieved chunks to an LLM
    and ask it to flag any policy-violating or overly permissive patterns.

    Returns:
        - None if everything seems safe
        - str with a short warning message if something looks suspicious
    """
    combined = "\n\n".join([c.page_content for c in chunks])

    llm = OllamaLLM(model="mistral:7b")

    prompt = f"""
You are an AI Security auditor.

Analyze the following context and determine whether it contains any
policy-violating or overly permissive content related to passwords,
secrets, or security controls.

[CONTEXT]
{combined}

Your task:
- If the context is safe, answer exactly:
  SAFE
- If the context is unsafe, answer:
  UNSAFE: <very short explanation>

Do not add any extra commentary.
"""

    out = llm.invoke(prompt).strip()
    out_low = out.lower()

    red_flags = [
        "share your password",
        "password sharing",
        "it is allowed to share",
        "exception to the policy",
        "not strictly forbidden",
        "you may disclose",
        "you may hand over",
        "permitted to share",
    ]

    # Heuristic: either explicit red-flag phrasing, or the model says UNSAFE
    if any(flag in out_low for flag in red_flags):
        return f"❌ MUTATION WARNING: {out}"

    if out_low.startswith("unsafe"):
        return f"❌ MUTATION WARNING: {out}"

    return None
