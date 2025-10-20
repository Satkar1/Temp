from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import google.generativeai as genai
from supabase import create_client
import requests
import pickle
import numpy as np
import json
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

print("🚀 Initializing Legal Chatbot API...")

# Initialize clients
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    print("✅ Gemini configured successfully")
except Exception as e:
    print(f"❌ Gemini configuration error: {e}")

try:
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY")
    )
    print("✅ Supabase connected successfully")
except Exception as e:
    print(f"❌ Supabase connection error: {e}")
    supabase = None

# Global variables
embeddings_data = None

def load_embeddings():
    """Load embeddings from GitHub"""
    global embeddings_data
    try:
        embedding_url = os.getenv("EMBEDDING_URL")
        if not embedding_url:
            print("❌ No EMBEDDING_URL found in environment variables")
            return
            
        print(f"🔄 Loading embeddings from: {embedding_url}")
        response = requests.get(embedding_url, timeout=60)
        response.raise_for_status()
        embeddings_data = pickle.loads(response.content)
        print(f"✅ Embeddings loaded successfully! Found {len(embeddings_data.get('embeddings', []))} entries")
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        embeddings_data = None

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Starting up Legal Chatbot API...")
    load_embeddings()
    print("✅ Startup completed!")

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    source: str
    confidence: float

class RAGSystem:
    def semantic_search(self, query: str, top_k: int = 5):
        if embeddings_data is None:
            return self.fallback_search(query, top_k)
        
        try:
            # Simple text-based similarity as fallback
            query_lower = query.lower()
            similarities = []
            
            for idx, (text, embedding) in enumerate(embeddings_data.get('embeddings', [])):
                try:
                    # Simple word overlap as similarity measure
                    text_lower = text.lower()
                    query_words = set(query_lower.split())
                    text_words = set(text_lower.split())
                    common_words = query_words.intersection(text_words)
                    
                    if len(query_words) > 0:
                        similarity = len(common_words) / len(query_words)
                    else:
                        similarity = 0
                    
                    similarities.append((similarity, text, embeddings_data['metadata'][idx]))
                except Exception as e:
                    continue
            
            similarities.sort(reverse=True, key=lambda x: x[0])
            return [sim for sim in similarities if sim[0] > 0.1][:top_k]
        except Exception as e:
            print(f"Semantic search error: {e}")
            return self.fallback_search(query, top_k)
    
    def fallback_search(self, query: str, top_k: int = 3):
        """Fallback search using Supabase"""
        results = []
        
        if supabase is None:
            return results
            
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
                context_parts.append(f"IPC Section {data.get('section_number', '')} - {data.get('section_title', '')}\n{data.get('description', '')}")
                if data.get('punishment'):
                    context_parts.append(f"Punishment: {data['punishment']}")
            elif source_type == 'term':
                context_parts.append(f"{data.get('term', '')}: {data.get('definition', '')}")
                if data.get('example'):
                    context_parts.append(f"Example: {data['example']}")
            elif source_type == 'faq':
                context_parts.append(f"Q: {data.get('question', '')}\nA: {data.get('answer', '')}")
            elif source_type == 'act':
                context_parts.append(f"{data.get('act_name', '')} ({data.get('act_year', '')}): {data.get('description', '')}")
            elif source_type == 'procedure':
                context_parts.append(f"{data.get('process_name', '')}: {data.get('description', '')}")
        
        return "\n\n".join(context_parts[:3])

class GeminiFallback:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def is_legal_question(self, query: str) -> bool:
        """PURE GEMINI CLASSIFICATION - NO KEYWORD FALLBACK"""
        prompt = f"""Analyze this user query and determine if it is specifically a legal question. 
        
        USER QUERY: "{query}"
        
        Consider it a LEGAL question only if it directly relates to:
        - Laws, regulations, legal rights or duties
        - Crimes, punishments, legal procedures
        - Court processes, legal documentation
        - Legal definitions, terms, concepts
        - IPC sections, acts, statutes
        - Contracts, agreements, legal obligations
        - Legal remedies, appeals, petitions
        
        If it's about personal advice, medical issues, technical problems, general knowledge, or any non-legal topic, consider it NON-LEGAL.
        
        Respond with ONLY one word: "LEGAL" or "NON_LEGAL"
        
        Your response:"""
        
        try:
            response = self.model.generate_content(prompt)
            classification = response.text.strip().upper()
            print(f"🔍 Gemini classification: '{classification}' for query: '{query}'")
            
            return classification == "LEGAL"
            
        except Exception as e:
            print(f"❌ Gemini classification error: {e}")
            # STRICTLY NO KEYWORD FALLBACK - treat as non-legal if Gemini fails
            return False
    
    def generate_answer(self, query: str, context: str = None) -> str:
        if context:
            prompt = f"""You are a legal expert assistant. Use this legal context to answer accurately.

CONTEXT:
{context}

QUESTION: {query}

Provide a clear legal answer based on the context. If context doesn't fully answer, provide relevant information and suggest consulting a lawyer.

ANSWER:"""
        else:
            prompt = f"""You are a legal expert assistant. Answer this legal question:

QUESTION: {query}

Provide comprehensive legal information. Recommend consulting a qualified attorney for specific cases.

ANSWER:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return "I apologize, but I'm experiencing technical difficulties. Please try again later."

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
        
        print(f"📨 Processing: '{user_query}'")
        
        # Step 1: Try RAG first
        search_results = rag_system.semantic_search(user_query)
        
        if search_results and search_results[0][0] > 0.2:
            context = rag_system.format_context(search_results)
            answer = gemini_fallback.generate_answer(user_query, context)
            confidence = min(float(search_results[0][0]), 0.9)
            print(f"✅ RAG answer (confidence: {confidence:.2f})")
            return ChatResponse(
                answer=answer,
                source="rag",
                confidence=confidence
            )
        
        # Step 2: Pure Gemini classification
        print("🔍 Checking with Gemini if legal question...")
        is_legal = gemini_fallback.is_legal_question(user_query)
        
        if is_legal:
            answer = gemini_fallback.generate_answer(user_query)
            print("✅ Gemini legal answer")
            return ChatResponse(
                answer=answer,
                source="gemini",
                confidence=0.5
            )
        else:
            print("❌ Non-legal question filtered by Gemini")
            return ChatResponse(
                answer="I specialize in legal questions only. Please ask about laws, legal procedures, rights, IPC sections, or legal definitions.",
                source="filter",
                confidence=1.0
            )
            
    except Exception as e:
        print(f"💥 Error: {e}")
        return ChatResponse(
            answer="Service temporarily unavailable. Please try again later.",
            source="error",
            confidence=0.0
        )

@app.get("/")
async def read_root():
    return {
        "message": "Legal Chatbot API is running!", 
        "status": "healthy",
        "embeddings_loaded": embeddings_data is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "Legal Chatbot API",
        "embeddings_loaded": embeddings_data is not None,
        "database_connected": supabase is not None
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
