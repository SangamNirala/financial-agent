"""Hugging Face integration for NLP and AI capabilities."""

import asyncio
from typing import Dict, Any, List, Optional
import aiohttp
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

from config.settings import settings, HUGGINGFACE_MODELS

class HuggingFaceClient:
    """Client for Hugging Face model interactions."""
    
    def __init__(self):
        self.api_key = settings.huggingface_api_key
        self.base_url = "https://api-inference.huggingface.co/models"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Local model cache
        self._local_models = {}
        self._tokenizers = {}
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def analyze_sentiment(
        self,
        text: str,
        model: str = "ProsusAI/finbert"
    ) -> Dict[str, Any]:
        """Analyze sentiment of financial text."""
        try:
            # Use local model if available, otherwise API
            if torch.cuda.is_available() or True:  # Prefer local for better control
                return await self._analyze_sentiment_local(text, model)
            else:
                return await self._analyze_sentiment_api(text, model)
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {"label": "neutral", "score": 0.5, "error": str(e)}
    
    async def _analyze_sentiment_local(self, text: str, model: str) -> Dict[str, Any]:
        """Analyze sentiment using local model."""
        try:
            # Load model if not cached
            if model not in self._local_models:
                sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model,
                    return_all_scores=True
                )
                self._local_models[model] = sentiment_pipeline
            
            classifier = self._local_models[model]
            
            # Run inference
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, classifier, text)
            
            # Process results
            if result and len(result) > 0:
                scores = result[0]
                # Find highest scoring label
                best_result = max(scores, key=lambda x: x['score'])
                
                return {
                    "label": best_result["label"].lower(),
                    "score": best_result["score"],
                    "all_scores": scores
                }
            
            return {"label": "neutral", "score": 0.5}
            
        except Exception as e:
            print(f"Local sentiment analysis error: {e}")
            return {"label": "neutral", "score": 0.5, "error": str(e)}
    
    async def _analyze_sentiment_api(self, text: str, model: str) -> Dict[str, Any]:
        """Analyze sentiment using Hugging Face API."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {"inputs": text}
        
        try:
            async with self.session.post(
                f"{self.base_url}/{model}",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result and len(result) > 0:
                        best_result = max(result, key=lambda x: x['score'])
                        return {
                            "label": best_result["label"].lower(),
                            "score": best_result["score"],
                            "all_scores": result
                        }
                else:
                    error_text = await response.text()
                    print(f"API error: {response.status} - {error_text}")
                    
        except Exception as e:
            print(f"API sentiment analysis error: {e}")
        
        return {"label": "neutral", "score": 0.5}
    
    async def generate_text(
        self,
        prompt: str,
        model: str = "microsoft/DialoGPT-large",
        max_length: int = 150,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate text using language model."""
        try:
            # Use local generation for better control
            return await self._generate_text_local(prompt, model, max_length, temperature)
        except Exception as e:
            print(f"Text generation error: {e}")
            return {"generated_text": "Text generation unavailable.", "error": str(e)}
    
    async def _generate_text_local(
        self,
        prompt: str,
        model: str,
        max_length: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Generate text using local model."""
        try:
            # Load model if not cached
            if model not in self._local_models:
                text_generator = pipeline(
                    "text-generation",
                    model=model,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=50256  # GPT-2 pad token
                )
                self._local_models[model] = text_generator
            
            generator = self._local_models[model]
            
            # Run inference
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: generator(prompt, max_length=max_length, num_return_sequences=1)
            )
            
            if result and len(result) > 0:
                generated_text = result[0]["generated_text"]
                # Remove the original prompt from the generated text
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                return {
                    "generated_text": generated_text,
                    "model": model,
                    "prompt_length": len(prompt)
                }
            
            return {"generated_text": "No text generated."}
            
        except Exception as e:
            print(f"Local text generation error: {e}")
            return {"generated_text": "Text generation failed.", "error": str(e)}
    
    async def summarize_text(
        self,
        text: str,
        model: str = "facebook/bart-large-cnn",
        max_length: int = 150,
        min_length: int = 50
    ) -> Dict[str, Any]:
        """Summarize text using summarization model."""
        try:
            # Load model if not cached
            if model not in self._local_models:
                summarizer = pipeline(
                    "summarization",
                    model=model,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                self._local_models[model] = summarizer
            
            summarizer = self._local_models[model]
            
            # Run inference
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: summarizer(text, max_length=max_length, min_length=min_length)
            )
            
            if result and len(result) > 0:
                return {
                    "summary": result[0]["summary_text"],
                    "model": model,
                    "original_length": len(text)
                }
            
            return {"summary": "Summarization failed."}
            
        except Exception as e:
            print(f"Summarization error: {e}")
            return {"summary": "Summarization unavailable.", "error": str(e)}
    
    async def classify_text(
        self,
        text: str,
        labels: List[str],
        model: str = "facebook/bart-large-mnli"
    ) -> Dict[str, Any]:
        """Classify text into given categories."""
        try:
            # Load model if not cached
            if model not in self._local_models:
                classifier = pipeline(
                    "zero-shot-classification",
                    model=model
                )
                self._local_models[model] = classifier
            
            classifier = self._local_models[model]
            
            # Run inference
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: classifier(text, labels)
            )
            
            if result:
                return {
                    "predicted_label": result["labels"][0],
                    "scores": dict(zip(result["labels"], result["scores"])),
                    "sequence": result["sequence"]
                }
            
            return {"predicted_label": labels[0], "scores": {}}
            
        except Exception as e:
            print(f"Classification error: {e}")
            return {"predicted_label": labels[0] if labels else "unknown", "error": str(e)}
    
    async def extract_entities(
        self,
        text: str,
        model: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    ) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        try:
            # Load model if not cached
            if model not in self._local_models:
                ner = pipeline(
                    "ner",
                    model=model,
                    aggregation_strategy="simple"
                )
                self._local_models[model] = ner
            
            ner = self._local_models[model]
            
            # Run inference
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, ner, text)
            
            return result if result else []
            
        except Exception as e:
            print(f"NER error: {e}")
            return []
    
    def clear_model_cache(self):
        """Clear cached models to free memory."""
        self._local_models.clear()
        self._tokenizers.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
