"""
    prompt with "seed" inside can find relevant example in seed_prompt.py
"""

QUERY_WO_HISTORY="""You will be given multiple reference documents, each begins with [Document ID]. 
Generate ONE natural-sounding question that:
1. Can be directly answered by ONLY ONE specific document
2. Sounds like a human question (don't mention the document)
3. Starts with the corresponding [Document ID]


Format: [Document ID] Your question here

Here is an example:

{SEED}

Here is the real user input:

**Documents:**
{DOCUMENTS}

"""



RESPONSE="""Based on the provided documents (and considering previous conversation, if applicable), think step-by-step and provide a detailed and complete answer to the user's question. Do not mention any document names or source information in your response.

**Documents:**
{DOCUMENTS}

**Question:**
{QUESTION}
"""



QUERY_W_HISTORY="""You will be given a conversation history and multiple reference documents, each beginning with [Document ID].

Generate one natural-sounding question that:
1. Can be directly answered by ONLY ONE specific document (don't mention the document).
2. **MUST ask about a DIFFERENT topic, aspect, or fact than ANY previous question.** Do NOT rephrase, paraphrase, or ask a variation of any prior question. Each new question should explore new information from an unused document.
3. Starts with the corresponding [Document ID].

Format: [Document ID] Your question here

Here is an example:

{SEED}

Here is the real user input:

**Previous Questions Asked (DO NOT repeat or rephrase any of these):**
{PREVIOUS_QUESTIONS}

**Conversation History:**
{HISTORY}

**Documents:**
{DOCUMENTS}

"""
