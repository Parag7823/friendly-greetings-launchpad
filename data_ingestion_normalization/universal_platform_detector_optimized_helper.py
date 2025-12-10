    async def _detect_platform_with_ai(self, payload: Dict, filename: str = None, user_id: str = None) -> Optional[Dict[str, Any]]:
        """Use AI to detect platform with enhanced prompting"""
        try:
            # Prepare comprehensive context for AI based on payload structure
            context = self._construct_detection_context(payload, filename)
            
            # Construct prompt for Groq Llama-3
            prompt = f"""
            Analyze the following data structure and identify the originating SaaS platform, ERP system, or bank.
            
            Input Data:
            {context[:4000]}  # Truncate to avoid token limits
            
            Task:
            1. Identify the specific platform (e.g., "Stripe", "QuickBooks", "Shopify", "Chase Bank", "Salesforce").
            2. Assign a confidence score (0.0 to 1.0).
            3. List specific indicators found (keys, values, formats).
            4. If unknown, output "unknown" with low confidence.
            
            Return ONLY a valid JSON object matching the requested schema.
            """
            
            # Call Groq API with instructor (Optimized)
            result = await self._safe_groq_call_with_instructor(prompt, temperature=0.1, max_tokens=256, user_id=user_id)
            return result
            
        except Exception as e:
            logger.error(f"AI platform detection failed: {e}")
            return None
