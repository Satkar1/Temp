from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# Global variable for embeddings
embeddings_data = None

# Load embeddings from GitHub
def load_embeddings():
    global embeddings_data
    try:
        embedding_url = os.getenv("EMBEDDING_URL", "https://raw.githubusercontent.com/yourusername/legal-embeddings/main/embedding.pkl")
        print(f"Loading embeddings from: {embedding_url}")
        response = requests.get(embedding_url)
        response.raise_for_status()
        embeddings_data = pickle.load(io.BytesIO(response.content))
        print("✅ Embeddings loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        embeddings_data = None

# Load embeddings on startup
@app.on_event("startup")
async def startup_event():
    load_embeddings()

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    source: str
    confidence: float

class RAGSystem:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def get_embedding(self, text: str):
        return self.model.encode(text)
    
    def semantic_search(self, query: str, top_k: int = 5):
        if embeddings_data is None:
            return self.fallback_search(query, top_k)
        
        try:
            query_embedding = self.get_embedding(query)
            
            # Calculate similarities
            similarities = []
            for idx, (text, embedding) in enumerate(embeddings_data.get('embeddings', [])):
                sim = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((sim, text, embeddings_data['metadata'][idx]))
            
            # Sort by similarity
            similarities.sort(reverse=True, key=lambda x: x[0])
            return similarities[:top_k]
        except Exception as e:
            print(f"Semantic search error: {e}")
            return self.fallback_search(query, top_k)
    
    def fallback_search(self, query: str, top_k: int = 3):
        """Fallback search using Supabase"""
        results = []
        
        searches = [
            ('ipc_sections', 'description', 'section'),
            ('legal_terms', 'definition', 'term'),
            ('faqs', 'question', 'faq'),
            ('acts', 'description', 'act'),
            ('procedures', 'description', 'procedure')
        ]
        
        for table, column, source_type in searches:
            try:
                response = supabase.table(table).select('*').ilike(column, f'%{query}%').limit(2).execute()
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
                context_parts.append(f"**IPC Section {data.get('section_number', '')} - {data.get('section_title', '')}**\n{data.get('description', '')}")
                if data.get('punishment'):
                    context_parts.append(f"**Punishment:** {data['punishment']}")
            elif source_type == 'term':
                context_parts.append(f"**{data.get('term', '')}**\n{data.get('definition', '')}")
                if data.get('example'):
                    context_parts.append(f"**Example:** {data['example']}")
            elif source_type == 'faq':
                context_parts.append(f"**Q:** {data.get('question', '')}\n**A:** {data.get('answer', '')}")
            elif source_type == 'act':
                context_parts.append(f"**{data.get('act_name', '')} ({data.get('act_year', '')})**\n{data.get('description', '')}")
            elif source_type == 'procedure':
                context_parts.append(f"**{data.get('process_name', '')}**\n{data.get('description', '')}")
        
        return "\n\n".join(context_parts[:5])

class GeminiFallback:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def is_legal_question(self, query: str) -> bool:
        prompt = f"""Analyze this query and respond with ONLY "YES" or "NO":
        
        Query: "{query}"
        
        Is this a legal question about: laws, rights, crimes, contracts, court procedures, legal definitions, IPC sections, legal terms, or legal procedures?
        
        Response: """
        
        try:
            response = self.model.generate_content(prompt)
            result = response.text.strip().upper()
            return "YES" in result
        except Exception as e:
            print(f"Gemini classification error: {e}")
            # Fallback: Check for legal keywords
            legal_keywords = ['law', 'legal', 'ipc', 'section', 'court', 'crime', 'right', 'act', 'contract', 'lawyer', 'judge', 'case', 'fir', 'bail', 'arrest']
            return any(keyword in query.lower() for keyword in legal_keywords)
    
    def generate_answer(self, query: str, context: str = None) -> str:
        if context:
            prompt = f"""You are a legal expert assistant. Use the following legal context to answer the user's question accurately.

LEGAL CONTEXT:
{context}

USER QUESTION: {query}

Provide a clear, accurate legal answer based on the context. If the context doesn't fully answer, provide the most relevant information available and suggest consulting a lawyer for specific cases.

ANSWER:"""
        else:
            prompt = f"""You are a legal expert assistant. Answer the following legal question helpfully and accurately.

QUESTION: {query}

Provide comprehensive legal information. Include relevant laws, procedures, or legal principles. Always recommend consulting with a qualified attorney for specific legal advice.

ANSWER:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return "I apologize, but I'm experiencing technical difficulties. Please try again in a few moments."

# Initialize systems
rag_system = RAGSystem()
gemini_fallback = GeminiFallback()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_query = request.message.strip()
        
        if not user_query:
            return ChatResponse(
                answer="Please enter a legal question.",
                source="error",
                confidence=0.0
            )
        
        print(f"Processing query: {user_query}")
        
        # Step 1: Try RAG with embeddings
        search_results = rag_system.semantic_search(user_query)
        
        if search_results and search_results[0][0] > 0.3:  # Confidence threshold
            context = rag_system.format_context(search_results)
            answer = gemini_fallback.generate_answer(user_query, context)
            confidence = min(float(search_results[0][0]), 0.95)
            return ChatResponse(
                answer=answer,
                source="rag",
                confidence=confidence
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
                answer="I specialize in legal questions only. Please ask about Indian laws, IPC sections, legal procedures, rights, or legal definitions.",
                source="filter",
                confidence=1.0
            )
            
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return ChatResponse(
            answer="Service temporarily unavailable. Please try again in a few moments.",
            source="error",
            confidence=0.0
        )

@app.get("/")
async def read_root():
    return {"message": "Legal Chatbot API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "Legal Chatbot API",
        "embeddings_loaded": embeddings_data is not None,
        "database_connected": supabase is not None
    }

@app.get("/test-embedding")
async def test_embedding():
    if embeddings_data:
        return {"status": "loaded", "samples": len(embeddings_data.get('embeddings', []))}
    else:
        return {"status": "not_loaded"}

# Serve frontend for testing
@app.get("/chat-ui")
async def chat_ui():
    return FileResponse('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
