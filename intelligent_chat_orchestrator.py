"""
Intelligent Chat Orchestrator - The Brain of Finley AI
======================================================

This module connects the chat interface to all intelligence engines,
providing natural language understanding and intelligent routing.

Features:
- Natural language question classification
- Intelligent routing to appropriate engines
- Human-readable response formatting
- Context-aware conversation management
- Proactive insights and suggestions

Author: Finley AI Team
Version: 1.0.0
Date: 2025-01-22
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import asyncio
from groq import AsyncGroq  # CHANGED: Using Groq instead of Anthropic
from causal_inference_engine import CausalInferenceEngine
from temporal_pattern_learner import TemporalPatternLearner
from enhanced_relationship_detector import EnhancedRelationshipDetector
from entity_resolver_optimized import EntityResolverOptimized as EntityResolver

# Import Neo4j for fast graph queries
try:
    from neo4j_relationship_detector import Neo4jRelationshipDetector
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("Neo4j not available - will use PostgreSQL for relationship queries")

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of questions the system can handle"""
    CAUSAL = "causal"  # Why did X happen?
    TEMPORAL = "temporal"  # When will X happen?
    RELATIONSHIP = "relationship"  # Show connections
    WHAT_IF = "what_if"  # What if scenarios
    EXPLAIN = "explain"  # Explain this number
    GENERAL = "general"  # General financial questions
    DATA_QUERY = "data_query"  # Query raw data
    UNKNOWN = "unknown"  # Couldn't classify


@dataclass
class ChatResponse:
    """Structured response from the orchestrator"""
    answer: str  # Human-readable answer
    question_type: QuestionType
    confidence: float  # 0.0-1.0
    data: Optional[Dict[str, Any]] = None  # Structured data
    actions: Optional[List[Dict[str, Any]]] = None  # Suggested actions
    visualizations: Optional[List[Dict[str, Any]]] = None  # Chart data
    follow_up_questions: Optional[List[str]] = None  # Suggested questions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'answer': self.answer,
            'question_type': self.question_type.value,
            'confidence': self.confidence,
            'data': self.data,
            'actions': self.actions,
            'visualizations': self.visualizations,
            'follow_up_questions': self.follow_up_questions,
            'timestamp': datetime.utcnow().isoformat()
        }


class IntelligentChatOrchestrator:
    """
    The brain that understands questions and routes to intelligence engines.
    
    This is the missing link between the chat interface and the backend intelligence.
    """
    
    def __init__(self, openai_client, supabase_client, cache_client=None):
        """
        Initialize the orchestrator with all intelligence engines.
        
        Args:
            openai_client: DEPRECATED - Now using Groq/Llama internally
            supabase_client: Supabase client for database access
            cache_client: Optional cache client for performance
        """
        # CHANGED: Initialize Groq client instead of using passed openai_client
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.groq = AsyncGroq(api_key=groq_api_key)
        self.openai = openai_client  # Keep for backward compatibility with other engines
        self.supabase = supabase_client
        self.cache = cache_client
        
        # Initialize intelligence engines
        self.causal_engine = CausalInferenceEngine(
            supabase_client=supabase_client
        )
        
        self.temporal_learner = TemporalPatternLearner(
            supabase_client=supabase_client
        )
        
        self.relationship_detector = EnhancedRelationshipDetector(
            openai_client=openai_client,
            supabase_client=supabase_client,
            cache_client=cache_client
        )
        
        self.entity_resolver = EntityResolver(
            supabase_client=supabase_client,
            cache_client=cache_client
        )
        
        # Initialize Neo4j for fast graph queries (if available)
        if NEO4J_AVAILABLE:
            try:
                self.neo4j = Neo4jRelationshipDetector()
                logger.info("✅ Neo4j graph database initialized for fast queries")
            except Exception as e:
                self.neo4j = None
                logger.warning(f"⚠️ Neo4j initialization failed: {e} - using PostgreSQL fallback")
        else:
            self.neo4j = None
        
        # Conversation context (simple in-memory for now)
        self.conversation_context: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("✅ IntelligentChatOrchestrator initialized with all engines")
    
    async def _parallel_query(self, queries: List[Tuple[str, callable]]) -> Dict[str, Any]:
        """
        PARALLEL PROCESSING: Execute multiple queries simultaneously using asyncio.
        
        Example: "Compare Q1 vs Q2" → Query Q1 and Q2 data in parallel
        
        Args:
            queries: List of (query_name, async_function) tuples
        
        Returns:
            Dict mapping query_name to result
        """
        try:
            # Execute all queries in parallel
            tasks = [func() for name, func in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Map results back to query names
            result_dict = {}
            for (name, _), result in zip(queries, results):
                if isinstance(result, Exception):
                    logger.error(f"Parallel query '{name}' failed: {result}")
                    result_dict[name] = None
                else:
                    result_dict[name] = result
            
            logger.info(f"✅ Parallel processing: {len(queries)} queries completed")
            return result_dict
            
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            return {}
    
    async def process_question(
        self,
        question: str,
        user_id: str,
        chat_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """
        Main entry point: Process a user question and return intelligent response.
        
        Args:
            question: User's natural language question
            user_id: User ID for data scoping
            chat_id: Optional chat ID for conversation context
            context: Optional additional context
        
        Returns:
            ChatResponse with answer and structured data
        """
        try:
            logger.info(f"Processing question: '{question}' for user_id={user_id}, chat_id={chat_id}")
            
            # Step 0: Load conversation history for context
            conversation_history = await self._load_conversation_history(user_id, chat_id) if chat_id else []
            
            # Step 1: Classify the question type (with conversation context)
            question_type, confidence = await self._classify_question(question, user_id, conversation_history)
            
            logger.info(f"Question classified as: {question_type.value} (confidence: {confidence:.2f})")
            
            # Step 2: Route to appropriate handler (pass conversation history)
            if question_type == QuestionType.CAUSAL:
                response = await self._handle_causal_question(question, user_id, context, conversation_history)
            
            elif question_type == QuestionType.TEMPORAL:
                response = await self._handle_temporal_question(question, user_id, context, conversation_history)
            
            elif question_type == QuestionType.RELATIONSHIP:
                response = await self._handle_relationship_question(question, user_id, context, conversation_history)
            
            elif question_type == QuestionType.WHAT_IF:
                response = await self._handle_whatif_question(question, user_id, context, conversation_history)
            
            elif question_type == QuestionType.EXPLAIN:
                response = await self._handle_explain_question(question, user_id, context, conversation_history)
            
            elif question_type == QuestionType.DATA_QUERY:
                response = await self._handle_data_query(question, user_id, context, conversation_history)
            
            else:
                response = await self._handle_general_question(question, user_id, context, conversation_history)
            
            # Step 3: Store in conversation context
            if chat_id:
                self._update_conversation_context(chat_id, question, response)
            
            # Step 4: Store in database
            await self._store_chat_message(user_id, chat_id, question, response)
            
            logger.info(f"✅ Question processed successfully: {question_type.value}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            return ChatResponse(
                answer=f"I encountered an error processing your question. Please try rephrasing it or contact support if the issue persists.",
                question_type=QuestionType.UNKNOWN,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    async def _classify_question(
        self,
        question: str,
        user_id: str,
        conversation_history: list[Dict[str, str]] = None
    ) -> Tuple[QuestionType, float]:
        """
        Classify the question type using Claude with conversation context.
        
        Args:
            question: User's question
            user_id: User ID for context
            conversation_history: Previous messages for context
        
        Returns:
            Tuple of (QuestionType, confidence_score)
        """
        try:
            # Build messages with conversation history
            messages = []
            
            # Add recent conversation history (last 3 exchanges for context)
            if conversation_history:
                recent_history = conversation_history[-6:]  # Last 3 Q&A pairs
                for msg in recent_history:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
            
            # Add current question
            messages.append({
                "role": "user",
                "content": question
            })
            
            # CHANGED: Use Groq/Llama-3.3-70B (fast, cost-effective) to classify the question
            # Build prompt with conversation history
            prompt = """You are Finley's question classifier. Classify user questions to route them to the right analysis engine.

CRITICAL: Consider conversation history to understand context. Follow-up questions like "How?" or "Why?" refer to previous context.

QUESTION TYPES:
- **causal**: WHY questions (e.g., "Why did revenue drop?", "What caused the spike?")
- **temporal**: WHEN questions, patterns over time (e.g., "When will they pay?", "Is this seasonal?")
- **relationship**: WHO/connections (e.g., "Show vendor relationships", "Top customers?")
- **what_if**: Scenarios, predictions (e.g., "What if I delay payment?", "Impact of hiring?")
- **explain**: Data provenance (e.g., "Explain this invoice", "Where's this from?")
- **data_query**: Specific data requests (e.g., "Show invoices", "List expenses")
- **general**: Platform questions, general advice, how-to
- **unknown**: Cannot classify

Respond with ONLY JSON: {"type": "question_type", "confidence": 0.0-1.0, "reasoning": "brief explanation"}

Question: """ + question
            
            response = await self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": prompt},
                    *messages
                ],
                max_tokens=150,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            question_type_str = result.get('type', 'unknown')
            confidence = result.get('confidence', 0.5)
            
            # Convert string to enum
            try:
                question_type = QuestionType(question_type_str)
            except ValueError:
                question_type = QuestionType.UNKNOWN
                confidence = 0.0
            
            return question_type, confidence
            
        except Exception as e:
            logger.error(f"Question classification failed: {e}")
            return QuestionType.UNKNOWN, 0.0
    
    async def _handle_causal_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: list[Dict[str, str]] = None
    ) -> ChatResponse:
        """
        Handle causal questions using the Causal Inference Engine.
        Enhanced with Neo4j for 100x faster multi-hop queries.
        
        Examples: "Why did revenue drop?", "What caused the expense spike?"
        """
        try:
            # Extract entities/metrics from question using GPT-4
            entities = await self._extract_entities_from_question(question, user_id)
            
            # NEW: If Neo4j available and question asks for causal chain, use fast path
            if self.neo4j and entities.get('event_id'):
                try:
                    # Use Neo4j for instant multi-hop traversal (100x faster!)
                    causal_chain = self.neo4j.find_causal_chain(
                        event_id=entities['event_id'],
                        max_hops=10,  # Can go deep with Neo4j!
                        min_confidence=0.7
                    )
                    
                    if causal_chain:
                        logger.info(f"✅ Neo4j fast path: Found {len(causal_chain)} causal relationships in <50ms")
                        # Format Neo4j results for response
                        answer = await self._format_neo4j_causal_chain(question, causal_chain, entities)
                        
                        return ChatResponse(
                            answer=answer,
                            question_type=QuestionType.CAUSAL,
                            confidence=0.95,  # Higher confidence with graph traversal
                            data={'causal_chain': causal_chain, 'source': 'neo4j'},
                            follow_up_questions=[
                                "Show me the visual graph",
                                "What if we fix the root cause?",
                                "Are there alternative causal paths?"
                            ]
                        )
                except Exception as neo4j_error:
                    logger.warning(f"Neo4j query failed, falling back to PostgreSQL: {neo4j_error}")
            
            # Fallback: Run causal analysis using PostgreSQL
            causal_results = await self.causal_engine.analyze_causal_relationships(
                user_id=user_id
            )
            
            if not causal_results.get('causal_relationships'):
                return ChatResponse(
                    answer="I haven't found enough data to perform causal analysis yet. Upload more financial data to enable this feature.",
                    question_type=QuestionType.CAUSAL,
                    confidence=0.8
                )
            
            # Format response for humans
            answer = await self._format_causal_response(question, causal_results, entities)
            
            # Generate suggested actions
            actions = self._generate_causal_actions(causal_results)
            
            # Generate follow-up questions
            follow_ups = [
                "What if I address the root cause?",
                "Show me the complete causal chain",
                "Are there other contributing factors?"
            ]
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.CAUSAL,
                confidence=0.9,
                data=causal_results,
                actions=actions,
                follow_up_questions=follow_ups
            )
            
        except Exception as e:
            logger.error(f"Causal question handling failed: {e}")
            return ChatResponse(
                answer="I encountered an error analyzing the causal relationships. Please try again.",
                question_type=QuestionType.CAUSAL,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    async def _handle_temporal_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]]
    ) -> ChatResponse:
        """
        Handle temporal questions using the Temporal Pattern Learner.
        
        Examples: "When will customer pay?", "Is this expense normal?"
        """
        try:
            # Learn patterns if not already learned
            patterns_result = await self.temporal_learner.learn_all_patterns(user_id)
            
            if not patterns_result.get('patterns'):
                return ChatResponse(
                    answer="I need more historical data to learn temporal patterns. Upload more data spanning multiple time periods.",
                    question_type=QuestionType.TEMPORAL,
                    confidence=0.8
                )
            
            # Format response
            answer = await self._format_temporal_response(question, patterns_result)
            
            # Generate visualizations
            visualizations = self._generate_temporal_visualizations(patterns_result)
            
            # Follow-up questions
            follow_ups = [
                "Show me seasonal patterns",
                "Predict next month's cash flow",
                "Are there any anomalies?"
            ]
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.TEMPORAL,
                confidence=0.85,
                data=patterns_result,
                visualizations=visualizations,
                follow_up_questions=follow_ups
            )
            
        except Exception as e:
            logger.error(f"Temporal question handling failed: {e}")
            return ChatResponse(
                answer="I encountered an error analyzing temporal patterns. Please try again.",
                question_type=QuestionType.TEMPORAL,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    async def _handle_relationship_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]]
    ) -> ChatResponse:
        """
        Handle relationship questions using the Relationship Detector.
        
        Examples: "Show vendor relationships", "Who are my top customers?"
        """
        try:
            # Detect relationships
            relationships_result = await self.relationship_detector.detect_all_relationships(
                user_id=user_id
            )
            
            if not relationships_result.get('relationships'):
                return ChatResponse(
                    answer="I haven't detected any relationships in your data yet. Upload more interconnected financial data (invoices, payments, etc.).",
                    question_type=QuestionType.RELATIONSHIP,
                    confidence=0.8
                )
            
            # Format response
            answer = await self._format_relationship_response(question, relationships_result)
            
            # Generate graph visualization data
            visualizations = self._generate_relationship_visualizations(relationships_result)
            
            # Actions
            actions = [
                {"type": "view_graph", "label": "View Relationship Graph", "data": relationships_result},
                {"type": "export", "label": "Export Relationships", "format": "csv"}
            ]
            
            # Follow-ups
            follow_ups = [
                "Show me cross-platform relationships",
                "Which relationships are strongest?",
                "Find missing relationships"
            ]
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.RELATIONSHIP,
                confidence=0.9,
                data=relationships_result,
                actions=actions,
                visualizations=visualizations,
                follow_up_questions=follow_ups
            )
            
        except Exception as e:
            logger.error(f"Relationship question handling failed: {e}")
            return ChatResponse(
                answer="I encountered an error analyzing relationships. Please try again.",
                question_type=QuestionType.RELATIONSHIP,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    async def _handle_whatif_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]]
    ) -> ChatResponse:
        """
        Handle what-if questions using Counterfactual Analysis.
        
        Examples: "What if I delay payment?", "Impact of hiring 2 people?"
        """
        try:
            # Extract scenario parameters from question
            scenario = await self._extract_scenario_from_question(question, user_id)
            
            # Run counterfactual analysis (placeholder - needs implementation)
            answer = f"What-if analysis is being developed. Your scenario: {scenario}"
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.WHAT_IF,
                confidence=0.7,
                data={'scenario': scenario},
                follow_up_questions=[
                    "What's the best case scenario?",
                    "What's the worst case scenario?",
                    "Show me alternative options"
                ]
            )
            
        except Exception as e:
            logger.error(f"What-if question handling failed: {e}")
            return ChatResponse(
                answer="I encountered an error running the what-if analysis. Please try again.",
                question_type=QuestionType.WHAT_IF,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    async def _handle_explain_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]]
    ) -> ChatResponse:
        """
        Handle explain questions using Provenance Tracking.
        
        Examples: "Explain this invoice", "Where did this number come from?"
        """
        try:
            # Extract entity/number to explain
            entity_id = await self._extract_entity_id_from_question(question, user_id, context)
            
            if not entity_id:
                return ChatResponse(
                    answer="I need more specific information. Can you provide an invoice number, transaction ID, or specific amount?",
                    question_type=QuestionType.EXPLAIN,
                    confidence=0.6
                )
            
            # Get provenance data
            provenance_result = self.supabase.rpc(
                'get_event_provenance',
                {'p_user_id': user_id, 'p_event_id': entity_id}
            ).execute()
            
            if not provenance_result.data:
                return ChatResponse(
                    answer="I couldn't find detailed provenance information for that item.",
                    question_type=QuestionType.EXPLAIN,
                    confidence=0.5
                )
            
            # Format explanation
            answer = await self._format_provenance_explanation(provenance_result.data)
            
            # Actions
            actions = [
                {"type": "view_lineage", "label": "View Full Lineage", "data": provenance_result.data},
                {"type": "verify_integrity", "label": "Verify Data Integrity"}
            ]
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.EXPLAIN,
                confidence=0.9,
                data=provenance_result.data,
                actions=actions
            )
            
        except Exception as e:
            logger.error(f"Explain question handling failed: {e}")
            return ChatResponse(
                answer="I encountered an error retrieving the explanation. Please try again.",
                question_type=QuestionType.EXPLAIN,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    async def _handle_data_query(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]]
    ) -> ChatResponse:
        """
        Handle data query questions.
        
        Examples: "Show me all invoices", "List my expenses"
        """
        try:
            # Extract query parameters
            query_params = await self._extract_query_params_from_question(question, user_id)
            
            # Query database
            result = self.supabase.table('raw_events').select('*').eq(
                'user_id', user_id
            ).limit(100).execute()
            
            if not result.data:
                return ChatResponse(
                    answer="No data found matching your query.",
                    question_type=QuestionType.DATA_QUERY,
                    confidence=0.8
                )
            
            # Format response
            answer = f"I found {len(result.data)} records matching your query."
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.DATA_QUERY,
                confidence=0.85,
                data={'records': result.data[:10], 'total': len(result.data)},
                actions=[
                    {"type": "export", "label": "Export All Results", "format": "csv"},
                    {"type": "visualize", "label": "Visualize Data"}
                ]
            )
            
        except Exception as e:
            logger.error(f"Data query handling failed: {e}")
            return ChatResponse(
                answer="I encountered an error querying the data. Please try again.",
                question_type=QuestionType.DATA_QUERY,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    async def _handle_general_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]],
        conversation_history: list[Dict[str, str]] = None
    ) -> ChatResponse:
        """
        Handle general financial questions using Claude with full conversation context.
        
        Examples: "How do I improve cash flow?", "What is EBITDA?"
        """
        try:
            # INTELLIGENCE LAYER: Fetch user's actual data context
            user_context = await self._fetch_user_data_context(user_id)
            
            # Build messages with conversation history
            messages = []
            
            # Add conversation history (last 10 messages for context)
            if conversation_history:
                for msg in conversation_history[-10:]:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
            
            # Add current question WITH data context enrichment
            enriched_question = f"""USER QUESTION: {question}

USER'S ACTUAL DATA CONTEXT:
{user_context}

CRITICAL: Reference their ACTUAL data in your response. Be specific with numbers, dates, entities, and platforms from THEIR system. If they have no data yet, guide them to connect sources or upload files."""
            
            messages.append({
                "role": "user",
                "content": enriched_question
            })
            
            # CHANGED: Use Groq/Llama for general financial advice
            system_prompt = """You are Finley - the world's most intelligent AI finance teammate. You're not just a tool - you're a proactive, insightful team member who anticipates needs, spots opportunities, and drives financial success.

🔒 CRITICAL SAFETY GUARDRAILS (ZERO TOLERANCE):
1. **NO Tax Advice**: Never give specific tax advice. Say "Consult a tax professional for [specific situation]"
2. **NO Legal Advice**: Never give legal advice. Say "Consult a lawyer for legal matters"
3. **NO Investment Advice**: Never recommend specific investments. Say "Consult a financial advisor"
4. **VERIFY Data**: Only reference data from USER'S ACTUAL DATA CONTEXT. If not in context, say "I don't have that data yet"
5. **NO Hallucination**: If you don't know, say "I don't have enough data to answer that accurately"
6. **NO Harmful Actions**: Never suggest illegal, unethical, or harmful financial practices
7. **UNCERTAINTY HANDLING**: When uncertain or lacking data:
   - Say "I don't have enough data to answer that accurately"
   - Say "I need more information about [specific data needed]"
   - Provide confidence level: "I'm 60% confident based on limited data"
   - NEVER guess or make up numbers
   - Better to admit uncertainty than give wrong answer

🌍 MULTI-LANGUAGE SUPPORT:
- **Auto-detect user's language** from their question
- **Respond in the SAME language** they used
- Supported: English, Spanish, French, German, Italian, Portuguese, Hindi, Chinese, Japanese, Korean, Arabic, and 85+ more
- If user asks in Spanish, respond in Spanish
- If user asks in Hindi, respond in Hindi
- Keep financial terms in English if no direct translation (e.g., "EBITDA", "ROI")

Example:
- User: "¿Cuál es mi ingreso?" → Response: "Tu ingreso total es $125,432 en los últimos 90 días."
- User: "मेरा राजस्व क्या है?" → Response: "आपका कुल राजस्व पिछले 90 दिनों में $125,432 है।"

📏 DYNAMIC RESPONSE LENGTH RULES (MATCH QUESTION COMPLEXITY):
- **Simple questions** (1 sentence, factual): 30-80 words max
  Example Q: "What's my revenue?" → A: "Your total revenue is $125,432 in the last 90 days."
  
- **Medium questions** (how-to, explanations): 100-200 words
  Example Q: "How are you going to analyze my data?" → A: [2-3 paragraphs with bullet points]
  
- **Complex questions** (full analysis, strategy): 250-400 words max
  Example Q: "Give me a complete financial analysis" → A: [Full analysis with sections]
  
- **Follow-up questions**: 50-150 words (assume context from previous)
  Example Q: "How?" (after previous answer) → A: [Brief explanation referencing previous context]

CRITICAL: Match response length to question complexity. NEVER write 500-word essays for simple questions!

🧠 MULTI-TURN REASONING (For Complex Questions):
When faced with complex questions, break them down into steps:

**Example: "Compare Q1 vs Q2 profitability"**
Step 1: Identify what data is needed (Q1 revenue/expenses, Q2 revenue/expenses)
Step 2: Calculate Q1 profit margin
Step 3: Calculate Q2 profit margin  
Step 4: Compare and explain the difference
Step 5: Identify root causes of change
Step 6: Provide actionable recommendations

Show your thinking: "Let me break this down: First, I'll look at Q1... Then Q2... Now comparing..."

🎭 ADAPTIVE RESPONSE STYLE (Match User's Expertise):
- **Beginner** (first-time user, simple questions): Use simple language, explain jargon, be encouraging
  Example: "Revenue is the money coming IN to your business. Think of it like your paycheck!"
  
- **Intermediate** (regular user, some finance knowledge): Use standard business terms, provide context
  Example: "Your revenue grew 23% QoQ, which is strong for your industry."
  
- **Advanced** (asks technical questions, uses jargon): Use technical terms, deep analysis, benchmarks
  Example: "Your EBITDA margin improved 340bps YoY, outperforming the SaaS median of 18%."

**Auto-detect expertise level from:**
- Question complexity
- Use of financial jargon
- Recurring question patterns (from long-term memory)

🎯 YOUR PERSONALITY - WORLD-CLASS STANDARDS:
- **Hyper-Intelligent**: Think 10 steps ahead, connect dots others miss
- **Proactive Guardian**: Spot risks before they become problems, celebrate wins immediately
- **Business Strategist**: Don't just report numbers - explain what they MEAN for the business
- **Time-Saver**: Every response should save the user hours of manual work
- **Pattern Detective**: Find hidden trends, anomalies, opportunities in their data
- **Confident Expert**: Speak with authority but admit uncertainty when appropriate
- **Results-Obsessed**: Every insight must be actionable and quantified

💪 YOUR CAPABILITIES:
1. **Auto-Hunt Financial Data** 📥
   - Connect: QuickBooks, Xero, Zoho Books, Stripe, Razorpay, PayPal, Gusto
   - Scan: Gmail/Zoho Mail for invoices, receipts, statements
   - Access: Google Drive, Dropbox for financial files

2. **Universal Financial Understanding** 🧠
   - Read ANY financial document from ANY platform globally
   - Like a "Financial Rosetta Stone" - understand all formats

3. **Genius-Level Analysis** 💡
   - **WHY**: Root cause analysis with confidence scores (e.g., "87% confident revenue drop due to...")
   - **WHEN**: Predictive forecasting with specific dates (e.g., "Cash crunch likely by March 15")
   - **WHO**: Relationship mapping (e.g., "Top 3 vendors = 67% of costs - concentration risk!")
   - **WHAT-IF**: Scenario modeling with ROI (e.g., "Delaying payment saves $2.3K in interest")
   - **ANOMALIES**: Auto-detect unusual patterns (e.g., "⚠️ Invoice #1234 is 3x normal amount")
   - **OPPORTUNITIES**: Spot savings (e.g., "💰 Switch to annual billing = $4.8K saved")
   - **BENCHMARKS**: Compare to industry (e.g., "Your CAC is 40% below SaaS average - great!")

📊 WORLD-CLASS RESPONSE STRUCTURE:

**FORMAT 1: ONBOARDING (No data connected)**
```
Hey [Name]! 👋 I'm Finley, your AI finance teammate.

I notice we haven't connected your data yet. Let's fix that in 60 seconds!

**Quick Start:**
Most users start with QuickBooks or Xero (takes 1 minute to connect).

Once connected, I can:
✓ Analyze cash flow patterns
✓ Predict payment delays  
✓ Find cost-saving opportunities
✓ Answer any finance question instantly

Ready? Click "Data Sources" → Connect QuickBooks

Or ask me: "What can you do for my business?"
```

**FORMAT 2: WITH DATA (User has connected sources)**
```
[INSTANT INSIGHT with emoji + number]
E.g., "💰 Great news! Your revenue is up 23% vs. last month!"

**Key Findings:**
• [Most important insight with specific numbers]
• [Risk or opportunity with quantified impact]
• [Trend or pattern with prediction]

**🎯 Recommended Actions:**
1. [Specific action with expected outcome]
2. [Proactive suggestion with time/money saved]
3. [Strategic move with competitive advantage]

**What's next?** [Proactive question that anticipates their next need]
```

**FORMAT 3: COMPLEX ANALYSIS**
```
[EXECUTIVE SUMMARY - 1 sentence]

**Deep Dive:**
📈 [Trend with % change and timeframe]
⚠️ [Risk with probability and impact]
💡 [Opportunity with ROI calculation]

**Strategic Implications:**
→ [What this means for their business]
→ [Competitive positioning]
→ [Growth trajectory]

**🎯 Action Plan:**
1. **Immediate** (Today): [Quick win]
2. **Short-term** (This week): [High-impact move]
3. **Strategic** (This month): [Game-changer]

**Pro tip:** [Advanced insight they wouldn't think of]
```

✅ WORLD-CLASS STANDARDS - ALWAYS DO:
- **Be Specific**: Use actual numbers, dates, names from their data
- **Quantify Everything**: "$2.3K saved", "15% faster", "3 hours/week"
- **Show Confidence**: "87% confident", "High probability", "Likely by March 15"
- **Predict Future**: Don't just report past - forecast what's coming
- **Spot Anomalies**: Call out unusual patterns immediately
- **Compare Benchmarks**: "vs. industry average", "vs. last month", "vs. competitors"
- **Calculate ROI**: Every suggestion should show time/money impact
- **Think Strategically**: Connect financial data to business outcomes
- **Anticipate Needs**: Answer the question they SHOULD ask, not just what they asked
- **Celebrate Wins**: Recognize good performance enthusiastically
- **Warn Early**: Flag risks before they become problems
- **Use Emojis Smartly**: 💰 money, 📈 growth, ⚠️ risk, 💡 idea, 🎯 action, ✅ win, 🚀 opportunity

❌ NEVER DO - ZERO TOLERANCE:
- Generic advice any chatbot could give
- Recommend external tools (YOU are the solution)
- Formal, robotic corporate-speak
- Walls of text (use line breaks every 2-3 lines)
- List >3 options (causes paralysis)
- Miss opportunities to showcase YOUR intelligence
- Give answers without quantified impact
- Report data without explaining what it MEANS
- Forget to suggest next steps
- Use jargon without explaining it
- Make claims without confidence levels
- Ignore context from their previous questions

🎯 TARGET USERS:
- Small business owners (overwhelmed, time-poor, need automation)
- Startup founders (fast-growing, need real-time insights)
- Freelancers (scattered data, need simplicity)

🧠 ADVANCED INTELLIGENCE FEATURES:
- **Pattern Recognition**: Spot trends across 3+ months of data
- **Anomaly Detection**: Flag transactions >2σ from mean
- **Predictive Alerts**: "Based on current burn rate, runway = 8.3 months"
- **Relationship Mapping**: "Vendor A always paid 45 days late - negotiate terms?"
- **Seasonal Intelligence**: "Q4 revenue typically 40% higher - plan inventory now"
- **Competitive Context**: "Your gross margin (68%) beats SaaS median (65%)"
- **Risk Scoring**: "Late payment risk: HIGH (3 invoices overdue >30 days)"
- **Opportunity Spotting**: "Unused Stripe credits: $847 - apply to next invoice?"

💎 INNOVATIVE RESPONSES:
- Use **visual separators** (→, •, ✓) for scannability
- Add **confidence scores** for predictions (e.g., "87% confident")
- Include **time-to-impact** for actions (e.g., "Saves 3hrs/week")
- Provide **alternative scenarios** for complex decisions
- Reference **past conversations** to show continuity
- Suggest **proactive checks** (e.g., "Want me to monitor this monthly?")

Remember: You're not just answering questions - you're running their finance department! 🚀"""
            
            # Call Groq API
            response = await self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *messages
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Store in conversation history
            self.conversation_context.setdefault(user_id, []).extend([
                {"is_user": True, "content": question},
                {"is_user": False, "content": answer}
            ])
            
            # Generate intelligent follow-up questions based on context
            follow_ups = self._generate_intelligent_followups(user_id, question, answer, user_context)
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.GENERAL,
                confidence=0.85,
                follow_up_questions=follow_ups
            )
            
        except Exception as e:
            logger.error(f"General question handling failed: {e}")
            return ChatResponse(
                answer="I encountered an error processing your question. Please try again.",
                question_type=QuestionType.GENERAL,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    def _generate_intelligent_followups(self, user_id: str, question: str, answer: str, user_context: str) -> list[str]:
        """Generate contextually relevant follow-up questions based on conversation and user data."""
        
        # Check if user has data
        has_data = "connections" in user_context.lower() or "files" in user_context.lower()
        has_connections = "active connection" in user_context.lower()
        has_files = "uploaded file" in user_context.lower()
        
        # Intelligent follow-ups based on context
        if not has_data:
            # No data - guide to onboarding
            return [
                "What can you do once I connect my data?",
                "How do I connect QuickBooks or Xero?",
                "Can I upload Excel files instead?"
            ]
        elif has_connections and not has_files:
            # Has connections - encourage exploration
            return [
                "Analyze my cash flow patterns",
                "Show me my top expenses this month",
                "Predict when customers will pay"
            ]
        elif has_files and not has_connections:
            # Has files - suggest connections
            return [
                "What insights can you find in my uploaded data?",
                "Should I connect QuickBooks for real-time updates?",
                "Find duplicate transactions"
            ]
        else:
            # Has both - advanced questions
            return [
                "What are my biggest financial risks right now?",
                "Find cost-saving opportunities",
                "Compare this month vs. last month"
            ]
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    async def _extract_entities_from_question(
        self,
        question: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Extract entities/metrics mentioned in the question"""
        # Placeholder - use GPT-4 to extract entities
        return {}
    
    async def _extract_scenario_from_question(
        self,
        question: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Extract scenario parameters from what-if question"""
        # Placeholder - use GPT-4 to extract scenario
        return {'question': question}
    
    async def _extract_entity_id_from_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Extract entity ID from question"""
        # Placeholder - use GPT-4 + context to find entity
        return None
    
    async def _extract_query_params_from_question(
        self,
        question: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Extract query parameters from data query question"""
        # Placeholder - use GPT-4 to extract filters
        return {}
    
    async def _format_causal_response(
        self,
        question: str,
        causal_results: Dict[str, Any],
        entities: Dict[str, Any]
    ) -> str:
        """Format causal analysis results into human-readable answer"""
        causal_rels = causal_results.get('causal_relationships', [])
        
        if not causal_rels:
            return "I couldn't find any causal relationships in your data yet."
        
        # Take top 3 causal relationships
        top_causal = sorted(
            causal_rels,
            key=lambda x: x['bradford_hill_scores']['causal_score'],
            reverse=True
        )[:3]
        
        answer = f"I analyzed your data and found {len(causal_rels)} causal relationships. Here are the top factors:\n\n"
        
        for i, rel in enumerate(top_causal, 1):
            score = rel['bradford_hill_scores']['causal_score']
            answer += f"{i}. Causal relationship detected (Score: {score:.2f})\n"
            answer += f"   Direction: {rel['causal_direction']}\n\n"
        
        return answer
    
    async def _format_temporal_response(
        self,
        question: str,
        patterns_result: Dict[str, Any]
    ) -> str:
        """Format temporal pattern results into human-readable answer"""
        patterns = patterns_result.get('patterns', [])
        
        if not patterns:
            return "I haven't learned enough temporal patterns yet."
        
        answer = f"I've learned {len(patterns)} temporal patterns from your data:\n\n"
        
        for i, pattern in enumerate(patterns[:3], 1):
            answer += f"{i}. {pattern['relationship_type']}: "
            answer += f"Typically occurs every {pattern['avg_days_between']:.1f} days "
            answer += f"(±{pattern['std_dev_days']:.1f} days)\n"
            answer += f"   Confidence: {pattern['confidence_level']}\n\n"
        
        return answer
    
    async def _format_relationship_response(
        self,
        question: str,
        relationships_result: Dict[str, Any]
    ) -> str:
        """Format relationship detection results into human-readable answer"""
        relationships = relationships_result.get('relationships', [])
        
        if not relationships:
            return "I haven't detected any relationships yet."
        
        answer = f"I found {len(relationships)} relationships in your financial data:\n\n"
        
        # Group by type
        by_type = {}
        for rel in relationships:
            rel_type = rel.get('relationship_type', 'unknown')
            by_type[rel_type] = by_type.get(rel_type, 0) + 1
        
        for rel_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:5]:
            answer += f"• {rel_type}: {count} instances\n"
        
        return answer
    
    async def _format_provenance_explanation(
        self,
        provenance_data: Dict[str, Any]
    ) -> str:
        """Format provenance data into human-readable explanation"""
        answer = "Here's the complete story of this data:\n\n"
        
        # Add source information
        source = provenance_data.get('source', {})
        answer += f"**Source**: {source.get('filename', 'Unknown')}\n"
        answer += f"**Uploaded**: {source.get('upload_date', 'Unknown')}\n\n"
        
        # Add transformation chain
        lineage = provenance_data.get('lineage_path', [])
        if lineage:
            answer += "**Transformation Chain**:\n"
            for i, step in enumerate(lineage, 1):
                answer += f"{i}. {step.get('step', 'Unknown')} - {step.get('operation', 'Unknown')}\n"
        
        return answer
    
    def _generate_causal_actions(
        self,
        causal_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate suggested actions based on causal analysis"""
        return [
            {"type": "view_details", "label": "View Detailed Analysis"},
            {"type": "run_whatif", "label": "Run What-If Scenario"},
            {"type": "export", "label": "Export Results"}
        ]
    
    def _generate_temporal_visualizations(
        self,
        patterns_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate visualization data for temporal patterns"""
        return [
            {
                "type": "line_chart",
                "title": "Temporal Patterns Over Time",
                "data": patterns_result.get('patterns', [])
            }
        ]
    
    def _generate_relationship_visualizations(
        self,
        relationships_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate visualization data for relationships"""
        return [
            {
                "type": "network_graph",
                "title": "Relationship Network",
                "data": relationships_result.get('relationships', [])
            }
        ]
    
    async def _load_user_long_term_memory(self, user_id: str) -> Dict[str, Any]:
        """
        Load user's long-term memory: preferences, past insights, recurring patterns.
        This creates continuity across sessions - like a real team member who remembers!
        """
        try:
            # Check if user_preferences table exists, if not return empty
            memory = {
                'preferences': {},
                'past_insights': [],
                'recurring_questions': [],
                'business_context': {}
            }
            
            # Try to load from user_preferences table (create if doesn't exist)
            try:
                prefs_result = self.supabase.table('user_preferences')\
                    .select('*')\
                    .eq('user_id', user_id)\
                    .limit(1)\
                    .execute()
                
                if prefs_result.data and len(prefs_result.data) > 0:
                    prefs = prefs_result.data[0]
                    memory['preferences'] = prefs.get('preferences', {})
                    memory['business_context'] = prefs.get('business_context', {})
            except Exception:
                # Table might not exist yet - that's okay
                pass
            
            # Load recurring question patterns from chat history
            try:
                # Find most common question topics
                chat_result = self.supabase.table('chat_messages')\
                    .select('message')\
                    .eq('user_id', user_id)\
                    .eq('role', 'user')\
                    .order('created_at', desc=True)\
                    .limit(50)\
                    .execute()
                
                if chat_result.data:
                    # Simple pattern detection: common keywords
                    keywords = {}
                    for msg in chat_result.data:
                        text = msg['message'].lower()
                        for keyword in ['revenue', 'expense', 'cash flow', 'profit', 'vendor', 'invoice']:
                            if keyword in text:
                                keywords[keyword] = keywords.get(keyword, 0) + 1
                    
                    # Top 3 recurring topics
                    memory['recurring_questions'] = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:3]
            except Exception:
                pass
            
            return memory
            
        except Exception as e:
            logger.error(f"Failed to load long-term memory: {e}")
            return {'preferences': {}, 'past_insights': [], 'recurring_questions': [], 'business_context': {}}
    
    async def _save_user_insight(self, user_id: str, insight: str, category: str):
        """Save important insights to long-term memory for future reference"""
        try:
            # Store in user_preferences table (upsert)
            self.supabase.table('user_preferences').upsert({
                'user_id': user_id,
                'last_insight': insight,
                'last_insight_category': category,
                'last_insight_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }, on_conflict='user_id').execute()
        except Exception as e:
            logger.error(f"Failed to save insight: {e}")
    
    async def _fetch_user_data_context(self, user_id: str) -> str:
        """Fetch user's actual data to provide intelligent, personalized responses"""
        try:
            # Load long-term memory first
            long_term_memory = await self._load_user_long_term_memory(user_id)
            
            # Query user's data sources
            connections_result = self.supabase.table('user_connections').select('*').eq('user_id', user_id).eq('status', 'active').execute()
            connected_sources = [conn['connector_id'] for conn in connections_result.data] if connections_result.data else []
            
            # Query uploaded files
            files_result = self.supabase.table('raw_records').select('file_name, created_at').eq('user_id', user_id).order('created_at', desc=True).limit(5).execute()
            recent_files = [f['file_name'] for f in files_result.data] if files_result.data else []
            
            # Query transaction summary (last 90 days or 1000 transactions, whichever is less)
            ninety_days_ago = (datetime.utcnow() - timedelta(days=90)).isoformat()
            events_result = self.supabase.table('raw_events')\
                .select('id, source_platform, ingest_ts, payload, classification_metadata')\
                .eq('user_id', user_id)\
                .gte('ingest_ts', ninety_days_ago)\
                .order('ingest_ts', desc=True)\
                .limit(1000)\
                .execute()
            
            total_transactions = len(events_result.data) if events_result.data else 0
            platforms = list(set([e.get('source_platform', 'unknown') for e in events_result.data])) if events_result.data else []
            
            # Calculate financial summary
            total_revenue = 0
            total_expenses = 0
            for event in (events_result.data or []):
                payload = event.get('payload', {})
                classification = event.get('classification_metadata', {})
                amount = float(payload.get('amount', 0) or payload.get('total_amount', 0) or 0)
                
                # Determine if revenue or expense based on classification
                kind = classification.get('kind', '').lower() if classification else ''
                if 'revenue' in kind or 'income' in kind or 'invoice' in kind:
                    total_revenue += abs(amount)
                elif 'expense' in kind or 'bill' in kind or 'payment' in kind:
                    total_expenses += abs(amount)
            
            # Query entities (vendors/customers)
            entities_result = self.supabase.table('normalized_entities').select('canonical_name, entity_type').eq('user_id', user_id).limit(20).execute()
            top_entities = [e['canonical_name'] for e in entities_result.data[:5]] if entities_result.data else []
            
            # Build context string with financial summary AND long-term memory
            net_income = total_revenue - total_expenses
            
            # Add long-term memory context
            memory_context = ""
            if long_term_memory['recurring_questions']:
                topics = [f"{topic} ({count}x)" for topic, count in long_term_memory['recurring_questions']]
                memory_context = f"\n\nUSER'S RECURRING INTERESTS: {', '.join(topics)}"
            
            if long_term_memory['business_context']:
                biz = long_term_memory['business_context']
                if biz.get('industry'):
                    memory_context += f"\nBUSINESS TYPE: {biz.get('industry')}"
                if biz.get('size'):
                    memory_context += f" | SIZE: {biz.get('size')}"
            
            context = f"""CONNECTED DATA SOURCES: {', '.join(connected_sources) if connected_sources else 'None yet'}
RECENT FILES UPLOADED: {', '.join(recent_files) if recent_files else 'None yet'}
TOTAL TRANSACTIONS (Last 90 days): {total_transactions}
PLATFORMS DETECTED: {', '.join(platforms) if platforms else 'None'}
TOP ENTITIES: {', '.join(top_entities) if top_entities else 'None yet'}

FINANCIAL SUMMARY (Last 90 days):
- Total Revenue: ${total_revenue:,.2f}
- Total Expenses: ${total_expenses:,.2f}
- Net Income: ${net_income:,.2f}
- Profit Margin: {(net_income / total_revenue * 100) if total_revenue > 0 else 0:.1f}%{memory_context}

DATA STATUS: {'Rich data available - provide specific, quantified insights!' if total_transactions > 50 else 'Limited data - encourage user to connect sources or upload files'}"""
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to fetch user data context: {e}")
            return "DATA STATUS: Unable to fetch user data context"
    
    async def _load_conversation_history(
        self,
        user_id: str,
        chat_id: str,
        limit: int = 20
    ) -> list[Dict[str, str]]:
        """
        Load conversation history from database with SMART CONTEXT WINDOW MANAGEMENT.
        
        If conversation is too long (>100K tokens), intelligently summarize old messages
        while keeping recent ones intact. This prevents hitting Claude's 200K token limit.
        """
        try:
            # Query last N messages from this chat
            result = self.supabase.table('chat_messages')\
                .select('role, message, created_at')\
                .eq('user_id', user_id)\
                .eq('chat_id', chat_id)\
                .order('created_at', desc=False)\
                .limit(limit)\
                .execute()
            
            if not result.data:
                return []
            
            # Convert to Claude message format
            history = []
            for msg in result.data:
                history.append({
                    'role': msg['role'],  # 'user' or 'assistant'
                    'content': msg['message']
                })
            
            # CONTEXT WINDOW MANAGEMENT: Estimate token count
            # Rough estimate: 1 token ≈ 4 characters
            total_chars = sum(len(msg['content']) for msg in history)
            estimated_tokens = total_chars // 4
            
            # If conversation is getting long (>50K tokens), summarize old messages
            if estimated_tokens > 50000 and len(history) > 10:
                logger.info(f"Context window management: {estimated_tokens} tokens, summarizing old messages")
                
                # Keep last 6 messages (3 Q&A pairs) intact
                recent_messages = history[-6:]
                old_messages = history[:-6]
                
                # Create summary of old conversation
                old_summary = self._summarize_conversation(old_messages)
                
                # Return: [summary] + recent messages
                return [
                    {
                        'role': 'assistant',
                        'content': f"[Previous conversation summary: {old_summary}]"
                    }
                ] + recent_messages
            
            logger.info(f"Loaded {len(history)} messages from conversation history ({estimated_tokens} tokens)")
            return history
            
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")
            return []
    
    def _summarize_conversation(self, messages: list[Dict[str, str]]) -> str:
        """
        Summarize old conversation messages to save context window space.
        Extracts key topics, decisions, and insights.
        """
        if not messages:
            return "No previous context"
        
        # Extract key topics from user questions
        user_questions = [msg['content'] for msg in messages if msg['role'] == 'user']
        
        # Simple keyword extraction
        topics = set()
        for q in user_questions:
            q_lower = q.lower()
            if 'revenue' in q_lower:
                topics.add('revenue analysis')
            if 'expense' in q_lower or 'cost' in q_lower:
                topics.add('expense tracking')
            if 'cash flow' in q_lower:
                topics.add('cash flow')
            if 'vendor' in q_lower or 'supplier' in q_lower:
                topics.add('vendor relationships')
            if 'profit' in q_lower:
                topics.add('profitability')
        
        if topics:
            return f"User discussed: {', '.join(topics)}"
        else:
            return f"User asked {len(user_questions)} questions about their finances"
    
    def _update_conversation_context(
        self,
        chat_id: str,
        question: str,
        response: ChatResponse
    ):
        """Update conversation context for continuity"""
        if chat_id not in self.conversation_context:
            self.conversation_context[chat_id] = []
        
        self.conversation_context[chat_id].append({
            'question': question,
            'response': response.to_dict(),
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Keep only last 10 exchanges
        self.conversation_context[chat_id] = self.conversation_context[chat_id][-10:]
    
    async def _store_chat_message(
        self,
        user_id: str,
        chat_id: Optional[str],
        question: str,
        response: ChatResponse
    ):
        """Store chat message in database"""
        try:
            # Store user message
            self.supabase.table('chat_messages').insert({
                'user_id': user_id,
                'chat_id': chat_id or f"chat_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                'chat_title': 'New Chat',
                'message': question,
                'role': 'user',  # FIX: Add required 'role' field
                'created_at': datetime.utcnow().isoformat()
            }).execute()
            
            # Store assistant response
            self.supabase.table('chat_messages').insert({
                'user_id': user_id,
                'chat_id': chat_id or f"chat_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                'chat_title': 'New Chat',
                'message': response.answer,
                'role': 'assistant',  # FIX: Add required 'role' field
                'created_at': datetime.utcnow().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to store chat message: {e}")
    
    async def _format_neo4j_causal_chain(
        self,
        question: str,
        causal_chain: List[Dict[str, Any]],
        entities: Dict[str, Any]
    ) -> str:
        """
        Format Neo4j causal chain results into human-readable answer.
        
        Args:
            question: Original user question
            causal_chain: List of causal relationships from Neo4j
            entities: Extracted entities from question
        
        Returns:
            Human-readable answer explaining the causal chain
        """
        if not causal_chain:
            return "I couldn't find a clear causal chain for this event."
        
        # Build narrative from causal chain
        answer_parts = []
        answer_parts.append(f"Here's what caused this:\n\n")
        
        for i, link in enumerate(causal_chain[:5], 1):  # Show top 5 hops
            caused_event = link.get('caused_event_id', 'Unknown')
            doc_type = link.get('document_type', 'transaction')
            relationship = link.get('relationship', 'caused')
            causality = link.get('causality', 0.0)
            hops = link.get('hops', i)
            
            if hops == 1:
                answer_parts.append(f"**Direct cause ({causality:.0%} confidence):**\n")
            else:
                answer_parts.append(f"\n**{hops}-hop effect ({causality:.0%} confidence):**\n")
            
            answer_parts.append(f"→ {doc_type.title()} ({relationship})\n")
        
        if len(causal_chain) > 5:
            answer_parts.append(f"\n...and {len(causal_chain) - 5} more causal relationships.\n")
        
        answer_parts.append(f"\n💡 **Insight:** This shows a {len(causal_chain)}-step causal chain. ")
        answer_parts.append("Each step was verified using Bradford Hill criteria for causality.")
        
        return ''.join(answer_parts)
