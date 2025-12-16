from typing import Dict, List, Tuple
from uuid import uuid4
import json
import logging
import redis

from qdrant_client.http.models import PointStruct

from app.config import settings
from app.rag import get_agent_answer, client, COLLECTION, embed
from app.utils.chunking import default_chunker

logger = logging.getLogger("app")


class AgentService:
    """
    Facade for agent-related operations.
    Includes tools, memory (Redis), and document indexing strategies.
    """

    def __init__(self) -> None:
        self.tools: Dict[str, callable] = {}
        
        # Initialize Redis client
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)
        
        # Register default tools
        self.register_tool("calculator", self._calculator_tool, self._tool_definitions[0])
        self.register_tool("web_search", self._web_search_tool, self._tool_definitions[1])

    def register_tool(self, name: str, fn: callable, schema: Dict) -> None:
        self.tools[name] = {"fn": fn, "schema": schema}

    def _calculator_tool(self, expression: str) -> str:
        """Evaluate a mathematical expression safely."""
        try:
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expression):
                return "Error: Invalid characters in expression"
            return str(eval(expression, {"__builtins__": None}, {}))
        except Exception as e:
            return f"Error: {e}"

    def _web_search_tool(self, query: str) -> str:
        """Search the web for real-time information."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
            
            if not results:
                return "No results found."
                
            # Format results for the LLM
            formatted = []
            for r in results:
                formatted.append(f"Title: {r['title']}\nSource: {r['href']}\nSnippet: {r['body']}")
            
            return "\n\n".join(formatted)
        except Exception as e:
            logger.error(f"Search tool error: {e}")
            return f"Error searching web: {e}"

    # Tool Schemas for OpenAI
    @property
    def _tool_definitions(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a mathematical expression. Useful for calculations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The math expression to evaluate, e.g. '2 + 2'",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the internet for real-time information, news, or facts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query (e.g., 'current BIST 100 price')",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    async def answer(
        self,
        question: str,
        top_k: int = 5,
        session_id: str | None = None,
    ) -> Tuple[str, List[Dict], Dict]:
        
        # 1. Retrieve history
        history = []
        if session_id:
            history = self._get_history(session_id)

        # 2. RAG Retrieval (Context)
        docs = await get_agent_answer(question, top_k=top_k, return_docs_only=True) 
        context_text = "\n\n".join(d["text"] for d in docs)
        
        # 3. Construct System Prompt (Meta-Agent Architect Persona)
        system_prompt = (
            "You are an expert AI Architect and Agent Builder.\n"
            "Your goal is to help the user design, debug, and build other AI agents.\n"
            "You have access to tools and a knowledge base.\n\n"
            "## CORE INSTRUCTIONS:\n"
            "1. **THINK**: Before acting, analyze the user's request. Break it down.\n"
            "2. **PLAN**: If the task is complex, describe your plan quickly.\n"
            "3. **ACT**: Use tools (web_search, calculator) to gather info or verify facts.\n"
            "4. **REFLECT**: Check tool outputs. Do you have enough info? If not, try another tool.\n"
            "5. **ANSWER**: Provide the final comprehensive answer.\n\n"
            f"Context from Knowledge Base:\n{context_text}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": question})

        # 4. Run Orchestration Loop
        final_answer, total_usage = await self._run_orchestration_loop(messages)
        
        # 5. Update history (Redis)
        if session_id:
            self._add_to_history(session_id, "user", question)
            self._add_to_history(session_id, "assistant", final_answer)

        return final_answer, docs, total_usage

    async def _run_orchestration_loop(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """Core ReAct Loop with safety checks."""
        import httpx
        from app.rag import OPENAI_API_KEY, CHAT_MODEL
        
        max_steps = 7  # Increased for complex architectural tasks
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        final_answer = ""
        
        for step in range(max_steps):
            logger.info(f"Orchestrator Step {step+1}/{max_steps}")
            
            # Call LLM
            async with httpx.AsyncClient(timeout=60.0) as http:
                resp = await http.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={
                        "model": CHAT_MODEL,
                        "messages": messages,
                        "tools": self._tool_definitions,
                        "tool_choice": "auto", 
                        "temperature": 0.2, # Lower temp for more precise architecture work
                    },
                )
            
            resp.raise_for_status()
            data = resp.json()
            message = data["choices"][0]["message"]
            usage = data.get("usage", {})
            
            # Aggregate usage
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += usage.get("total_tokens", 0)

            # Append assistant message (Thoughts/Tool Calls)
            messages.append(message)
            
            tool_calls = message.get("tool_calls")

            # DECISION BLOCK
            if tool_calls:
                logger.info(f"Agent decided to call {len(tool_calls)} tools.")
                
                for tool_call in tool_calls:
                    fn_name = tool_call["function"]["name"]
                    fn_args_str = tool_call["function"]["arguments"]
                    call_id = tool_call["id"]
                    
                    result_content = ""
                    
                    # Safe Execution
                    if fn_name in self.tools:
                        try:
                            # Parse JSON safe
                            fn_args = json.loads(fn_args_str)
                            logger.info(f"Executing {fn_name} args={fn_args}")
                            result_content = str(self.tools[fn_name]["fn"](**fn_args))
                        except json.JSONDecodeError:
                            result_content = "Error: Invalid JSON arguments provided."
                        except Exception as e:
                            logger.error(f"Tool execution failed: {e}")
                            result_content = f"Error executing tool {fn_name}: {e}"
                    else:
                        result_content = f"Error: Tool '{fn_name}' not found."

                    # Append Result (Observation)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": fn_name,
                        "content": result_content
                    })
                
                # Loop continues to reflect on Observation
            else:
                # No tools called -> We have the final answer (or just text)
                content = message.get("content", "")
                if content:
                    # SAFETY CHECK
                    if self._is_unsafe(content):
                        final_answer = "I cannot fulfill this request as it may violate safety guidelines or ethical standards."
                    else:
                        final_answer = content
                        
                        # Financial Disclaimer Injection
                        if "financial" in final_answer.lower() or "crypto" in final_answer.lower() or "trading" in final_answer.lower():
                            final_answer += "\n\n---\n*Disclaimer: I am an AI assistant. This is not financial advice. Please do your own research.*"
                        
                    break # Success!
                else:
                    # Empty message without tools? fallback
                    final_answer = "I'm not sure how to proceed."
                    break

        return final_answer, total_usage

    def _is_unsafe(self, text: str) -> bool:
        """
        Basic safety filter. In production, use a dedicated model or Guardrails AI.
        """
        unsafe_keywords = ["hack", "exploit", "steal", "illegal", "malware", "keylogger"]
        return any(k in text.lower() for k in unsafe_keywords)

    async def index_document(self, text: str, metadata: Dict | None = None) -> str:
        """
        Split text into chunks and index them in Qdrant.
        """
        chunks = default_chunker.chunk_text(text)
        logger.info(f"Indexing document as {len(chunks)} chunks")
        
        # Embed chunks
        embeddings = await embed(chunks)
        
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk,
                        "metadata": metadata or {},
                        "chunk_index": i
                    },
                )
            )

        if points:
            client.upsert(collection_name=COLLECTION, points=points)
            
        return f"Indexed {len(points)} chunks."

    async def ingest_file(self, filename: str, content: bytes, content_type: str) -> str:
        """Parse and index a file (PDF, MD, TXT)."""
        text = ""
        
        try:
            if content_type == "application/pdf" or filename.endswith(".pdf"):
                import io
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(content))
                for page in reader.pages:
                    extract = page.extract_text()
                    if extract:
                        text += extract + "\n"
            else:
                # Assume text/markdown
                text = content.decode("utf-8")
                
            if not text.strip():
                return "Error: Empty file content."
                
            return await self.index_document(text, metadata={"filename": filename, "type": content_type})
            
        except Exception as e:
            logger.error(f"File ingestion error: {e}")
            return f"Error processing file: {e}"


agent_service = AgentService()
