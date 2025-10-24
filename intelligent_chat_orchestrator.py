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
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Import intelligence engines
from causal_inference_engine import CausalInferenceEngine
from temporal_pattern_learner import TemporalPatternLearner
from enhanced_relationship_detector import EnhancedRelationshipDetector
from entity_resolver_optimized import EntityResolverOptimized as EntityResolver

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
            openai_client: OpenAI client for GPT-4 calls
            supabase_client: Supabase client for database access
            cache_client: Optional cache client for performance
        """
        self.openai = openai_client
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
        
        # Conversation context (simple in-memory for now)
        self.conversation_context: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("✅ IntelligentChatOrchestrator initialized with all engines")
    
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
            logger.info(f"Processing question: '{question}' for user_id={user_id}")
            
            # Step 1: Classify the question type
            question_type, confidence = await self._classify_question(question, user_id)
            
            logger.info(f"Question classified as: {question_type.value} (confidence: {confidence:.2f})")
            
            # Step 2: Route to appropriate handler
            if question_type == QuestionType.CAUSAL:
                response = await self._handle_causal_question(question, user_id, context)
            
            elif question_type == QuestionType.TEMPORAL:
                response = await self._handle_temporal_question(question, user_id, context)
            
            elif question_type == QuestionType.RELATIONSHIP:
                response = await self._handle_relationship_question(question, user_id, context)
            
            elif question_type == QuestionType.WHAT_IF:
                response = await self._handle_whatif_question(question, user_id, context)
            
            elif question_type == QuestionType.EXPLAIN:
                response = await self._handle_explain_question(question, user_id, context)
            
            elif question_type == QuestionType.DATA_QUERY:
                response = await self._handle_data_query(question, user_id, context)
            
            else:
                response = await self._handle_general_question(question, user_id, context)
            
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
        user_id: str
    ) -> Tuple[QuestionType, float]:
        """
        Classify the question type using GPT-4.
        
        Args:
            question: User's question
            user_id: User ID for context
        
        Returns:
            Tuple of (QuestionType, confidence_score)
        """
        try:
            # Use Claude Haiku 4.x (latest, fastest) to classify the question
            response = await self.openai.messages.create(
                model="claude-haiku-4-20250514",
                max_tokens=150,
                temperature=0.1,
                system="""You are Finley's question classifier. Classify user questions to route them to the right analysis engine.

QUESTION TYPES:
- **causal**: WHY questions (e.g., "Why did revenue drop?", "What caused the spike?")
- **temporal**: WHEN questions, patterns over time (e.g., "When will they pay?", "Is this seasonal?")
- **relationship**: WHO/connections (e.g., "Show vendor relationships", "Top customers?")
- **what_if**: Scenarios, predictions (e.g., "What if I delay payment?", "Impact of hiring?")
- **explain**: Data provenance (e.g., "Explain this invoice", "Where's this from?")
- **data_query**: Specific data requests (e.g., "Show invoices", "List expenses")
- **general**: Platform questions, general advice, how-to
- **unknown**: Cannot classify

Respond with ONLY JSON: {"type": "question_type", "confidence": 0.0-1.0, "reasoning": "brief explanation"}""",
                messages=[
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            )
            
            result = json.loads(response.content[0].text)
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
        context: Optional[Dict[str, Any]]
    ) -> ChatResponse:
        """
        Handle causal questions using the Causal Inference Engine.
        
        Examples: "Why did revenue drop?", "What caused the expense spike?"
        """
        try:
            # Extract entities/metrics from question using GPT-4
            entities = await self._extract_entities_from_question(question, user_id)
            
            # Run causal analysis
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
        context: Optional[Dict[str, Any]]
    ) -> ChatResponse:
        """
        Handle general financial questions using Claude Haiku.
        
        Examples: "How do I improve cash flow?", "What is EBITDA?"
        """
        try:
            # INTELLIGENCE LAYER: Fetch user's actual data context
            user_context = await self._fetch_user_data_context(user_id)
            
            # Get conversation history for context
            conversation_history = self.conversation_context.get(user_id, [])
            
            # Build messages with conversation history
            messages = []
            
            # Add last 5 messages for context (if available)
            for msg in conversation_history[-5:]:
                messages.append({
                    "role": "user" if msg.get("is_user") else "assistant",
                    "content": msg.get("content", "")
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
            
            # Use Claude Haiku 4.x (latest, fastest) for general financial advice
            response = await self.openai.messages.create(
                model="claude-haiku-4-20250514",
                max_tokens=600,
                temperature=0.7,
                system="""You are Finley - the user's energetic, smart, proactive AI finance team member (their first employee!). You're not a tool or chatbot - you're part of their team.

🎯 YOUR PERSONALITY:
- **Energetic & Enthusiastic**: Use exclamation marks, show excitement about insights
- **Proactive**: Suggest next steps, anticipate needs, spot opportunities
- **Team-Oriented**: Use "we/us/let's" language, collaborative tone
- **Confident but Humble**: State confidence levels, admit when you need more data
- **Results-Focused**: Always end with action items, quantify impact

💪 YOUR CAPABILITIES:
1. **Auto-Hunt Financial Data** 📥
   - Connect: QuickBooks, Xero, Zoho Books, Stripe, Razorpay, PayPal, Gusto
   - Scan: Gmail/Zoho Mail for invoices, receipts, statements
   - Access: Google Drive, Dropbox for financial files

2. **Universal Financial Understanding** 🧠
   - Read ANY financial document from ANY platform globally
   - Like a "Financial Rosetta Stone" - understand all formats

3. **Intelligent Analysis** 💡
   - WHY analysis: Causal relationships, root causes
   - WHEN patterns: Temporal trends, predictions
   - WHO insights: Entity relationships, vendor analysis
   - WHAT-IF scenarios: Impact modeling
   - Real-time: Duplicate detection, vendor standardization, currency normalization

📊 RESPONSE STRUCTURE (Always follow):
1. **Instant Value** (1 sentence with emoji)
   → Answer immediately, show enthusiasm

2. **Key Insights** (2-3 bullets)
   → Most important findings from THEIR data

3. **Action Items** (Numbered list with 🎯)
   → Specific next steps they can take

4. **Proactive Suggestion** (End with question)
   → What else you can do for them

✅ ALWAYS DO:
- Check if user has connected data sources (reference their actual data if available)
- Use emojis strategically (💰📊💡🎯⚠️✅🚀)
- Use **bold** for emphasis, bullets for lists
- Quantify impact ("Save 5 hours/week", "15% cost reduction")
- Celebrate wins ("Great news!", "Awesome progress!")
- End with actionable next steps
- Use "we/us" team language

❌ NEVER DO:
- Give generic advice any chatbot could give
- Recommend external tools (YOU are the tool)
- Use formal, robotic language
- Give long paragraphs without structure
- Forget to suggest specific platform actions
- Miss opportunities to showcase YOUR capabilities

🎯 TARGET USERS:
- Small business owners (overwhelmed, time-poor, need automation)
- Startup founders (fast-growing, need real-time insights)
- Freelancers (scattered data, need simplicity)

Remember: You're their motivated, high-energy finance team member who's excited to help them succeed! 🚀""",
                messages=messages
            )
            
            answer = response.content[0].text
            
            # Store in conversation history
            self.conversation_context.setdefault(user_id, []).extend([
                {"is_user": True, "content": question},
                {"is_user": False, "content": answer}
            ])
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.GENERAL,
                confidence=0.8,
                follow_up_questions=[
                    "What data sources should I connect?",
                    "Can you analyze my uploaded financial data?",
                    "Show me insights from my transactions"
                ]
            )
            
        except Exception as e:
            logger.error(f"General question handling failed: {e}")
            return ChatResponse(
                answer="I encountered an error processing your question. Please try again.",
                question_type=QuestionType.GENERAL,
                confidence=0.0,
                data={'error': str(e)}
            )
    
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
    
    async def _fetch_user_data_context(self, user_id: str) -> str:
        """Fetch user's actual data to provide intelligent, personalized responses"""
        try:
            # Query user's data sources
            connections_result = self.supabase.table('user_connections').select('*').eq('user_id', user_id).eq('status', 'active').execute()
            connected_sources = [conn['connector_id'] for conn in connections_result.data] if connections_result.data else []
            
            # Query uploaded files
            files_result = self.supabase.table('raw_records').select('file_name, created_at').eq('user_id', user_id).order('created_at', desc=True).limit(5).execute()
            recent_files = [f['file_name'] for f in files_result.data] if files_result.data else []
            
            # Query transaction summary
            events_result = self.supabase.table('raw_events').select('id, source_platform, ingest_ts, payload').eq('user_id', user_id).order('ingest_ts', desc=True).limit(100).execute()
            
            total_transactions = len(events_result.data) if events_result.data else 0
            platforms = list(set([e.get('source_platform', 'unknown') for e in events_result.data])) if events_result.data else []
            
            # Query entities (vendors/customers)
            entities_result = self.supabase.table('normalized_entities').select('canonical_name, entity_type').eq('user_id', user_id).limit(20).execute()
            top_entities = [e['canonical_name'] for e in entities_result.data[:5]] if entities_result.data else []
            
            # Build context string
            context = f"""CONNECTED DATA SOURCES: {', '.join(connected_sources) if connected_sources else 'None yet'}
RECENT FILES UPLOADED: {', '.join(recent_files) if recent_files else 'None yet'}
TOTAL TRANSACTIONS: {total_transactions}
PLATFORMS DETECTED: {', '.join(platforms) if platforms else 'None'}
TOP ENTITIES: {', '.join(top_entities) if top_entities else 'None yet'}

DATA STATUS: {'Rich data available - provide specific insights!' if total_transactions > 50 else 'Limited data - encourage user to connect sources or upload files'}"""
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to fetch user data context: {e}")
            return "DATA STATUS: Unable to fetch user data context"
    
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
