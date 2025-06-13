from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

class LLMConfig(BaseModel):
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000

class LLMProcessor:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Define prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""Based on the following context, please answer the query. 
            If the answer cannot be found in the context, say so.

            Context:
            {context}

            Query: {query}

            Answer:"""
        )
        
        # Form the LLM chain
        self.chain = self.prompt_template | self.llm

    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate response using LLM."""
        # Combine context documents
        combined_context = "\n\n".join([doc['text'] for doc in context])
        
        # Generate response
        response = self.chain.invoke({"context": combined_context, "query": query})
        
        return response