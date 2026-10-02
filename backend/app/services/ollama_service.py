import logging
import httpx
from typing import Dict, Any, Optional
from ..config import settings
from ..utils.exceptions import ResearchGuardException

logger = logging.getLogger(__name__)

class OllamaServiceError(ResearchGuardException):
    """Exception raised for errors during Ollama API calls."""
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message=message, status_code=status_code)

class OllamaService:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Sends a prompt to the local Ollama HTTP API (/api/generate) and returns the generated text response.
        Does not require any cloud API key or external network calls.
        """
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else settings.OLLAMA_TEMPERATURE,
                "num_ctx": settings.OLLAMA_NUM_CTX,
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    error_msg = f"Ollama returned HTTP {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    raise OllamaServiceError(message=error_msg, status_code=503)

                data = response.json()
                generated_text = data.get("response", "").strip()
                eval_duration_ns = data.get("eval_duration", 0)
                eval_duration_ms = eval_duration_ns / 1_000_000.0 if eval_duration_ns else 0.0

                return {
                    "response": generated_text,
                    "model": data.get("model", self.model),
                    "done": data.get("done", True),
                    "eval_duration_ms": eval_duration_ms,
                    "total_duration_ms": data.get("total_duration", 0) / 1_000_000.0 if data.get("total_duration") else 0.0,
                }
        except httpx.TimeoutException as err:
            error_msg = f"Ollama request timed out after {timeout} seconds."
            logger.error(error_msg)
            raise OllamaServiceError(message=error_msg, status_code=504)
        except httpx.RequestError as err:
            error_msg = f"Failed to connect to local Ollama server at {self.base_url}: {err}"
            logger.error(error_msg)
            raise OllamaServiceError(message=error_msg, status_code=503)
        except OllamaServiceError:
            raise
        except Exception as err:
            error_msg = f"Unexpected error while communicating with Ollama: {err}"
            logger.error(error_msg)
            raise OllamaServiceError(message=error_msg, status_code=500)


ollama_service = OllamaService()
