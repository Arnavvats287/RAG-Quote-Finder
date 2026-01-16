import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

class Generator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2
        )

    def generate_answer(self, query, retrieved):
        """
        Generates a concise natural language answer
        using retrieved quote context.
        """

        if not retrieved:
            return "No relevant quotes found for your query."

        context = "\n".join(
            [
                f"- \"{r['quote']}\" — {r['author']}"
                for r in retrieved
            ]
        )

        prompt = f"""
You are an assistant answering questions using quotes as context.

Context:
{context}

Question:
{query}

Instructions:
- Answer concisely (1–2 sentences)
- Stay grounded in the provided quotes
- Do NOT hallucinate new quotes

Answer:
"""

        response = self.llm.invoke(prompt)
        return response.content.strip()
