import datetime
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple
from src.document_handler.document_retrieval import DocumentRetrieval

# Updated Prompt Template for Context-Aware RAG Bot
PROMPT_TEMPLATE = """
Today is {today}. Use the provided context to answer the user's question accurately. Ensure that key details such as name, date of birth, passport number, and place of birth are considered when relevant.

[START]
## Query History:
{query_history}

## User Query:
{user_input}

## Retrieved Context:
{context}

## Additional Context Extraction:
- If the user asks about personal details like name, date of birth, passport number, or place of birth, extract it from the context.
- If the question relates to an address, place of issue, or nationality, prioritize those details.
- If no direct answer is found, provide the closest relevant information.

## Assistant Thought:
{assistant_thought}

## Assistant Response:
{assistant_response}
[END]
"""


class BotAssistant(BaseModel):
    llm: Any
    prompt_template: str = PROMPT_TEMPLATE
    # Stores (user input, AI response)
    query_history: List[Tuple[str, str]] = []
    contexts: List[Dict[str, Any]] = []
    verbose: bool = False
    threshold: float = 0.5

    class Config:  # Use this for Pydantic V1
        arbitrary_types_allowed = True

    def run(self, query: str) -> str:
        # Log user input
        document_retrieval = DocumentRetrieval()

        # Query Pinecone or OCR-extracted text
        matches = document_retrieval.query_index(query)

        if matches:
            top_contexts = [match["metadata"]["text"]
                            for match in matches if match["score"] >= self.threshold]
            context_score = matches[0]["score"]
            context_thought = "The retrieved context has relevant details about the user."
        else:
            top_contexts = ["NO CONTEXT FOUND"]
            context_score = 0
            context_thought = "No relevant context was found."

        # Construct query history
        formatted_query = "\n".join(
            [f"User: {user}\nAssistant: {response}" for user,
                response in self.query_history[-5:]]
        )  # Keeps last 5 interactions

        # Prepare prompt with structured context extraction logic
        prompt = self.prompt_template.format(
            today=datetime.date.today(),
            query_history=formatted_query if formatted_query else "No previous context available.",
            user_input=query,
            context="\n".join(top_contexts),
            context_score=context_score,
            assistant_thought=context_thought,
            assistant_response="",
        )

        # Generate response
        response = self.llm.generate(prompt, stop=["[END]"])

        # Maintain query history
        self.query_history.append((query, response))

        return response
