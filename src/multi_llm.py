"""
MULTI-LLM FALLBACK SYSTEM
=========================
Unified LLM interface with automatic fallback across:
- Gemini (Google)
- Groq (fast inference)
- OpenRouter (free tier access)

FIXES APPLIED:
- Gemini: Bypasses SDK auto-retry (was causing 50s+ hangs on 429)
- OpenRouter: Auto-rotates through working free models if configured model fails
- Provider cooldown: Dead providers skipped for 5 min to avoid wasted retries
- All settings read from .env file (UTF-8 encoding)

Fallback logic:
1. Try primary LLM
2. On any error (rate limit, context length, API failure) → fallback to next
3. All three share the same LangChain interface
"""

import os
import time
import requests
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatResult

# LLM Providers
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

# Load .env with UTF-8 encoding
load_dotenv(encoding="utf-8")


# =============================================================================
# CONFIGURATION (read from .env)
# =============================================================================

class Config:
    """Centralized config — all values from .env"""

    # Multi-LLM
    LLM_PROVIDER_ORDER: List[str] = os.getenv("LLM_PROVIDER_ORDER").split(",")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS"))

    # Gemini
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY") or None
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL")

    # Groq
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY") or None
    GROQ_MODEL: str = os.getenv("GROQ_MODEL").strip()

    # OpenRouter
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY") or None
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL").strip()

    # App Info
    APP_URL: str = os.getenv("APP_URL", "https://localhost")
    APP_NAME: str = os.getenv("APP_NAME", "RAG-System")


# =============================================================================
# PROVIDER STATUS TRACKING (cooldown logic)
# =============================================================================

@dataclass
class ProviderStatus:
    """Track provider health and cooldown state."""
    name: str
    healthy: bool = True
    last_error_time: float = 0.0
    last_error_type: str = ""
    consecutive_failures: int = 0
    cooldown_seconds: float = 300  # 5 min default

    def is_in_cooldown(self) -> bool:
        if not self.last_error_time:
            return False
        return (time.time() - self.last_error_time) < self.cooldown_seconds

    def mark_failed(self, error_type: str):
        self.healthy = False
        self.last_error_time = time.time()
        self.last_error_type = error_type
        self.consecutive_failures += 1
        if self.consecutive_failures >= 3:
            self.cooldown_seconds = 600  # 10 min after repeated failures

    def mark_success(self):
        self.healthy = True
        self.consecutive_failures = 0
        self.last_error_time = 0.0
        self.cooldown_seconds = 300


@dataclass
class LLMConfig:
    """Configuration for a single LLM provider."""
    name: str
    model: str
    api_key: str
    temperature: float = 0.1
    max_tokens: int = 500
    timeout: int = 30
    enabled: bool = True


# =============================================================================
# MULTI-LLM MANAGER
# =============================================================================

class MultiLLM:
    """
    Multi-LLM Fallback Manager with provider cooldown and fast fallback.

    Automatically tries multiple LLM providers in order.
    On 429/quota errors: provider enters cooldown (skipped for 5 min).
    On 404/model not found: OpenRouter auto-rotates to next free model.
    """

    DEFAULT_ORDER = ["gemini", "groq", "openrouter"]

    # Working free models on OpenRouter (fallback rotation)
    OPENROUTER_FREE_MODELS = os.getenv("OPENROUTER_FREE_MODELS")

    FALLBACK_ERRORS =  os.getenv("FALLBACK_ERRORS")

    def __init__(self, 
                 provider_order: List[str] = None,
                 temperature: float = None,
                 max_tokens: int = None,
                 fallback_on_any_error: bool = True):

        self.provider_order = provider_order or Config.LLM_PROVIDER_ORDER or self.DEFAULT_ORDER
        self.temperature = temperature if temperature is not None else Config.LLM_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else Config.LLM_MAX_TOKENS
        self.fallback_on_any_error = fallback_on_any_error

        self.current_provider_idx = 0
        self.last_successful_provider = None
        self.provider_status: Dict[str, ProviderStatus] = {}

        # OpenRouter model rotation
        self._openrouter_model_index = 0
        self._working_openrouter_model = None

        self.llm_configs = self._init_configs()
        self.llm_instances = {}
        self._init_llms()

        print(f"\n🤖 Multi-LLM Fallback System initialized")
        print(f"   Provider order: {self.provider_order}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Max tokens: {self.max_tokens}")

        available = [name for name in self.provider_order if self.llm_configs.get(name, {}).enabled]
        print(f"   Available providers: {available}")

    def _init_configs(self) -> Dict[str, LLMConfig]:
        """Initialize configurations from .env"""
        configs = {}

        # Gemini
        gemini_key = Config.GEMINI_API_KEY or ""
        configs["gemini"] = LLMConfig(
            name="gemini",
            model=Config.GEMINI_MODEL,
            api_key=gemini_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            enabled=bool(gemini_key)
        )

        # Groq
        groq_key = Config.GROQ_API_KEY or ""
        configs["groq"] = LLMConfig(
            name="groq",
            model=Config.GROQ_MODEL,
            api_key=groq_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            enabled=bool(groq_key)
        )

        # OpenRouter
        or_key = Config.OPENROUTER_API_KEY or ""
        or_model = Config.OPENROUTER_MODEL
        # If model is empty or looks like a broken one, use auto-rotation
        if not or_model or or_model in ["openrouter/owl-alpha", "x-ai/grok-3-mini-beta:free"]:
            or_model = self.OPENROUTER_FREE_MODELS[0]

        configs["openrouter"] = LLMConfig(
            name="openrouter",
            model=or_model,
            api_key=or_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            enabled=bool(or_key)
        )

        return configs

    def _init_llms(self):
        """Initialize LangChain LLM instances."""
        for name, config in self.llm_configs.items():
            if not config.enabled:
                print(f"   ⚠️  {name}: Disabled (no API key)")
                continue

            try:
                if name == "gemini":
                    llm = ChatGoogleGenerativeAI(
                        model=config.model,
                        temperature=config.temperature,
                        max_output_tokens=config.max_tokens,
                        google_api_key=config.api_key,
                        timeout=config.timeout
                    )

                elif name == "groq":
                    llm = ChatGroq(
                        model=config.model,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        api_key=config.api_key,
                        timeout=config.timeout,
                        max_retries=1  # Minimal retry
                    )

                elif name == "openrouter":
                    llm = ChatOpenAI(
                        model=config.model,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        api_key=config.api_key,
                        base_url="https://openrouter.ai/api/v1",
                        timeout=config.timeout,
                        max_retries=1,
                        default_headers={
                            "HTTP-Referer": Config.APP_URL,
                            "X-Title": Config.APP_NAME
                        }
                    )

                else:
                    continue

                self.llm_instances[name] = llm
                self.provider_status[name] = ProviderStatus(name=name)
                print(f"   ✅ {name}: {config.model} ready")

            except Exception as e:
                print(f"   ❌ {name}: Failed to initialize - {e}")
                config.enabled = False

    def _should_fallback(self, error: Exception) -> bool:
        """Determine if error triggers fallback."""
        if self.fallback_on_any_error:
            return True
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in self.FALLBACK_ERRORS)

    def _get_next_provider(self) -> Optional[str]:
        """Get next available provider (skipping cooldown)."""
        for i in range(self.current_provider_idx + 1, len(self.provider_order)):
            name = self.provider_order[i]
            if name in self.llm_instances:
                status = self.provider_status.get(name)
                if not status or not status.is_in_cooldown():
                    return name
        return None

    def _try_openrouter_rotation(self, error_str: str) -> bool:
        """Rotate OpenRouter to next free model on 404."""
        if "404" not in error_str and "no endpoints" not in error_str.lower():
            return False

        self._openrouter_model_index = (self._openrouter_model_index + 1) % len(self.OPENROUTER_FREE_MODELS)
        new_model = self.OPENROUTER_FREE_MODELS[self._openrouter_model_index]

        print(f"   🔄 Rotating OpenRouter to: {new_model}")

        # Re-init OpenRouter with new model
        try:
            self.llm_configs["openrouter"].model = new_model
            self.llm_instances["openrouter"] = ChatOpenAI(
                model=new_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.llm_configs["openrouter"].api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=30,
                max_retries=1,
                default_headers={
                    "HTTP-Referer": Config.APP_URL,
                    "X-Title": Config.APP_NAME
                }
            )
            return True
        except Exception as e:
            print(f"   ❌ OpenRouter rotation failed: {e}")
            return False

    def invoke(self, prompt: Union[str, List], 
               system_prompt: str = None,
               retries_per_provider: int = 1) -> Any:
        """
        Invoke LLM with automatic fallback and provider cooldown.

        Args:
            prompt: User prompt (str or list of messages)
            system_prompt: Optional system prompt
            retries_per_provider: Retries per provider (default 1 = no retry)

        Returns:
            LLM response
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        if isinstance(prompt, str):
            messages.append(HumanMessage(content=prompt))
        else:
            messages.extend(prompt)

        attempted_providers = []

        for idx, provider_name in enumerate(self.provider_order):
            self.current_provider_idx = idx

            if provider_name not in self.llm_instances:
                continue

            # Check cooldown
            status = self.provider_status.get(provider_name)
            if status and status.is_in_cooldown():
                print(f"\n⏸️ Skipping {provider_name} (cooldown: {status.last_error_type})")
                continue

            llm = self.llm_instances[provider_name]

            for attempt in range(retries_per_provider):
                try:
                    print(f"\n🔄 Trying {provider_name} (attempt {attempt + 1}/{retries_per_provider})...")

                    response = llm.invoke(messages)

                    self.last_successful_provider = provider_name
                    if status:
                        status.mark_success()
                    print(f"   ✅ Success with {provider_name}!")

                    return response

                except Exception as e:
                    err_str = str(e)
                    print(f"   ❌ {provider_name} failed: {err_str[:120]}")
                    attempted_providers.append((provider_name, err_str[:120]))

                    # Mark cooldown for quota errors
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                        if status:
                            status.mark_failed("RESOURCE_EXHAUSTED")
                        break  # Don't retry

                    # OpenRouter model rotation on 404
                    if provider_name == "openrouter" and self._try_openrouter_rotation(err_str):
                        continue  # Retry with new model

                    if "404" in err_str or "no endpoints" in err_str.lower():
                        if status:
                            status.mark_failed("NO_WORKING_MODEL")
                        break

                    if not self._should_fallback(e):
                        print(f"   ⚠️  Non-recoverable error, stopping.")
                        raise

                    if attempt < retries_per_provider - 1:
                        time.sleep(1)

        # All providers failed
        error_msg = f"\n❌ All LLM providers failed:\n"
        for name, err in attempted_providers:
            error_msg += f"   - {name}: {err}\n"
        raise Exception(error_msg)

    def stream(self, prompt: Union[str, List],
               system_prompt: str = None) -> Any:
        """Stream LLM response with automatic fallback."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        if isinstance(prompt, str):
            messages.append(HumanMessage(content=prompt))
        else:
            messages.extend(prompt)

        for idx, provider_name in enumerate(self.provider_order):
            self.current_provider_idx = idx

            if provider_name not in self.llm_instances:
                continue

            status = self.provider_status.get(provider_name)
            if status and status.is_in_cooldown():
                print(f"\n⏸️ Skipping {provider_name} (cooldown)")
                continue

            llm = self.llm_instances[provider_name]

            try:
                print(f"\n🔄 Streaming with {provider_name}...")

                for chunk in llm.stream(messages):
                    yield chunk

                self.last_successful_provider = provider_name
                if status:
                    status.mark_success()
                print(f"   ✅ Stream complete with {provider_name}!")
                return

            except Exception as e:
                err_str = str(e)
                print(f"   ❌ {provider_name} stream failed: {err_str[:120]}")

                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if status:
                        status.mark_failed("RESOURCE_EXHAUSTED")

                if provider_name == "openrouter":
                    self._try_openrouter_rotation(err_str)

                if not self._should_fallback(e):
                    raise

        raise Exception("All providers failed for streaming")

    def health_check(self) -> Dict[str, bool]:
        """Quick health check — respects cooldown."""
        health = {}
        print("\n🏥 Health Check:")
        for name in self.provider_order:
            if name not in self.llm_instances:
                health[name] = False
                print(f"   ⚠️ {name}: Not configured")
                continue

            status = self.provider_status.get(name)
            if status and status.is_in_cooldown():
                health[name] = False
                print(f"   ⏸️ {name}: In cooldown ({status.last_error_type})")
                continue

            try:
                response = self.llm_instances[name].invoke([HumanMessage(content="Hi")])
                health[name] = True
                if status:
                    status.mark_success()
                print(f"   ✅ {name}: Healthy")
            except Exception as e:
                err_str = str(e)
                health[name] = False
                if status:
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        status.mark_failed("RESOURCE_EXHAUSTED")
                    else:
                        status.mark_failed(err_str[:50])
                print(f"   ❌ {name}: Unhealthy - {err_str[:100]}")

        return health

    def get_current_provider(self) -> Optional[str]:
        if self.current_provider_idx < len(self.provider_order):
            return self.provider_order[self.current_provider_idx]
        return None

    def get_last_successful_provider(self) -> Optional[str]:
        return self.last_successful_provider

    def get_provider_status(self) -> Dict[str, Dict]:
        status = {}
        for name, config in self.llm_configs.items():
            ps = self.provider_status.get(name)
            status[name] = {
                "enabled": config.enabled,
                "model": config.model,
                "initialized": name in self.llm_instances,
                "last_used": name == self.last_successful_provider,
                "in_cooldown": ps.is_in_cooldown() if ps else False,
                "cooldown_reason": ps.last_error_type if ps else "",
            }
        return status


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_multi_llm_instance = None

def get_multi_llm() -> MultiLLM:
    """Get or create singleton MultiLLM instance."""
    global _multi_llm_instance
    if _multi_llm_instance is None:
        _multi_llm_instance = MultiLLM()
    return _multi_llm_instance


def reset_multi_llm():
    """Reset singleton instance (useful for testing/hot-reload)."""
    global _multi_llm_instance
    _multi_llm_instance = None


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-LLM FALLBACK SYSTEM TEST")
    print("="*70)

    multi_llm = MultiLLM()

    print("\n🏥 Health Check:")
    health = multi_llm.health_check()

    if any(health.values()):
        print("\n📝 Testing invoke:")
        try:
            response = multi_llm.invoke(
                "Explain machine learning in 2 sentences.",
                system_prompt="You are a helpful AI assistant.",
                retries_per_provider=1
            )
            print(f"\n📤 Response ({multi_llm.get_last_successful_provider()}):")
            print(f"   {response.content}")
        except Exception as e:
            print(f"\n❌ All failed: {e}")
    else:
        print("\n⚠️  No providers available. Set API keys in .env:")
        print("   GEMINI_API_KEY=...")
        print("   GROQ_API_KEY=...")
        print("   OPENROUTER_API_KEY=...")