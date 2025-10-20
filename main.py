from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import google.generativeai as genai
from supabase import create_client
import requests
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import io
from typing import List, Optional
import uvicorn

app = FastAPI(title="Legal Chatbot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load precomputed embeddings from GitHub
def load_embeddings_from_url():
    embedding_url = "https://raw.githubusercontent.com/yourusername/your-repo/main/embedding.pkl"
    try:
        response = requests.get(embedding_url)
        response.raise_for_status()
        embeddings_data = pickle.load(io.BytesIO(response.content))
        return embeddings_data
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

# Global variable for embeddings
embeddings_data = load_embeddings_from_url()

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    source: str
    confidence: float

class RAGSystem:
    def get_embedding(self, text: str):
        return embedding_model.encode(text)
    
    def semantic_search(self, query: str, top_k: int = 5):
        if embeddings_data is None:
            return self.fallback_search(query, top_k)
        
        query_embedding = self.get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for idx, (text, embedding) in enumerate(embeddings_data['embeddings']):
            sim = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((sim, text, embeddings_data['metadata'][idx]))
        
        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])
        return similarities[:top_k]
    
    def fallback_search(self, query: str, top_k: int = 3):
        """Fallback search using Supabase full-text search"""
        results = []
        
        # Search across all tables
        searches = [
            ('ipc_sections', 'description', 'section'),
            ('legal_terms', 'definition', 'term'),
            ('faqs', 'question', 'faq'),
            ('acts', 'description', 'act'),
            ('procedures', 'description', 'procedure')
        ]
        
        for table, column, source_type in searches:
            try:
                response = supabase.table(table).select('*').text_search(column, query).limit(2).execute()
                for item in response.data:
                    text = f"{source_type}: {item.get(column, '')}"
                    results.append((0.7, text, {'type': source_type, 'data': item}))
            except Exception as e:
                print(f"Search error in {table}: {e}")
        
        return results
    
    def format_context(self, search_results):
        context_parts = []
        for score, text, metadata in search_results:
            data = metadata['data']
            source_type = metadata['type']
            
            if source_type == 'section':
                context_parts.append(f"IPC Section {data.get('section_number', '')}: {data.get('description', '')}")
                if data.get('punishment'):
                    context_parts.append(f"Punishment: {data['punishment']}")
            elif source_type == 'term':
                context_parts.append(f"Legal Term: {data.get('term', '')} - {data.get('definition', '')}")
            elif source_type == 'faq':
                context_parts.append(f"Q: {data.get('question', '')}\nA: {data.get('answer', '')}")
            elif source_type == 'act':
                context_parts.append(f"Act: {data.get('act_name', '')} - {data.get('description', '')}")
            elif source_type == 'procedure':
                context_parts.append(f"Procedure: {data.get('process_name', '')} - {data.get('description', '')}")
        
        return "\n\n".join(context_parts[:5])  # Limit to top 5

class GeminiFallback:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def is_legal_question(self, query: str) -> bool:
        prompt = f"""Analyze this query and respond with ONLY "YES" or "NO":
        
        Query: "{query}"
        
        Is this a legal question about laws, rights, crimes, contracts, procedures, or legal definitions?
        Response: """
        
        try:
            response = self.model.generate_content(prompt)
            return "YES" in response.text.strip().upper()
        except:
            return False
    
    def generate_answer(self, query: str, context: str = None) -> str:
        if context:
            prompt = f"""Based on this legal context:
            {context}
            
            Question: {query}
            
            Provide a helpful legal answer. If context doesn't fully answer, provide general legal guidance.
            Answer: """
        else:
            prompt = f"""As a legal expert, answer this legal question:
            {query}
            
            Provide comprehensive legal information and suggest consulting a lawyer for specific cases.
            Answer: """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I apologize, but I'm unable to provide an answer right now. Please try again later."

# Initialize systems
rag_system = RAGSystem()
gemini_fallback = GeminiFallback()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_query = request.message.strip()
        
        # Step 1: Try RAG with embeddings
        search_results = rag_system.semantic_search(user_query)
        
        if search_results and search_results[0][0] > 0.3:  # Confidence threshold
            context = rag_system.format_context(search_results)
            answer = gemini_fallback.generate_answer(user_query, context)
            return ChatResponse(
                answer=answer,
                source="rag",
                confidence=float(search_results[0][0])
            )
        
        # Step 2: Gemini fallback
        is_legal = gemini_fallback.is_legal_question(user_query)
        
        if is_legal:
            answer = gemini_fallback.generate_answer(user_query)
            return ChatResponse(
                answer=answer,
                source="gemini",
                confidence=0.6
            )
        else:
            return ChatResponse(
                answer="I specialize in legal questions only. Please ask about laws, legal procedures, rights, or legal definitions.",
                source="filter",
                confidence=1.0
            )
            
    except Exception as e:
        return ChatResponse(
            answer="Service temporarily unavailable. Please try again later.",
            source="error",
            confidence=0.0
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "Legal Chatbot API",
        "embeddings_loaded": embeddings_data is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
