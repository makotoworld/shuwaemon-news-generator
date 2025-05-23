"""
複数LLMプロバイダー対応の統合クライアント
"""
import os
from typing import Tuple, Optional, Dict, Any, Literal
from abc import ABC, abstractmethod

# LLMプロバイダーの型定義
LLMProvider = Literal["gemini", "openai", "anthropic"]

class BaseLLMClient(ABC):
    """LLMクライアントの基底クラス"""
    
    @abstractmethod
    def generate_article(self, prompt: str, temperature: float = 0.7) -> Tuple[str, int, int, float]:
        """記事を生成し、トークン数とコストを返す"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """APIが利用可能かチェック"""
        pass
    
    @abstractmethod
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """トークン数からコストを計算"""
        pass

class GeminiClient(BaseLLMClient):
    """Google Gemini APIクライアント"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model_name = "gemini-2.0-flash-lite"
        self._initialize()
    
    def _initialize(self):
        """Gemini APIの初期化"""
        try:
            if self.api_key:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.genai = genai
                self._available = True
            else:
                self._available = False
        except ImportError:
            print("警告: google-generativeai パッケージがインストールされていません")
            self._available = False
        except Exception as e:
            print(f"Gemini API初期化エラー: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        return self._available
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Geminiのコスト計算"""
        input_cost_per_1k = 0.0003
        output_cost_per_1k = 0.0006
        return (input_tokens / 1000) * input_cost_per_1k + (output_tokens / 1000) * output_cost_per_1k
    
    def generate_article(self, prompt: str, temperature: float = 0.7) -> Tuple[str, int, int, float]:
        """Gemini APIで記事を生成"""
        if not self.is_available():
            raise Exception("Gemini APIが利用できません")
        
        try:
            generation_config = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
            
            model = self.genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt, generation_config=generation_config)
            
            # トークン数の概算
            input_tokens = len(prompt.split())
            output_tokens = len(response.text.split())
            
            # コスト計算（Gemini 2.0 Flash Lite）
            input_cost_per_1k = 0.0003
            output_cost_per_1k = 0.0006
            cost = (input_tokens / 1000) * input_cost_per_1k + (output_tokens / 1000) * output_cost_per_1k
            
            return response.text, input_tokens, output_tokens, cost
        except Exception as e:
            raise Exception(f"Gemini記事生成エラー: {e}")

class OpenAIClient(BaseLLMClient):
    """OpenAI APIクライアント"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = "gpt-4o-mini"  # コスト効率の良いモデル
        self._initialize()
    
    def _initialize(self):
        """OpenAI APIの初期化"""
        try:
            if self.api_key:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                self._available = True
            else:
                self._available = False
        except ImportError:
            print("警告: openai パッケージがインストールされていません")
            self._available = False
        except Exception as e:
            print(f"OpenAI API初期化エラー: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        return self._available
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """OpenAIのコスト計算"""
        input_cost_per_1k = 0.00015
        output_cost_per_1k = 0.0006
        return (input_tokens / 1000) * input_cost_per_1k + (output_tokens / 1000) * output_cost_per_1k
    
    def generate_article(self, prompt: str, temperature: float = 0.7) -> Tuple[str, int, int, float]:
        """OpenAI APIで記事を生成"""
        if not self.is_available():
            raise Exception("OpenAI APIが利用できません")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "あなたは専門的なニュース記事ライターです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=4096
            )
            
            article = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            # コスト計算（GPT-4o-mini）
            input_cost_per_1k = 0.00015  # $0.00015 per 1K input tokens
            output_cost_per_1k = 0.0006  # $0.0006 per 1K output tokens
            cost = (input_tokens / 1000) * input_cost_per_1k + (output_tokens / 1000) * output_cost_per_1k
            
            return article, input_tokens, output_tokens, cost
        except Exception as e:
            raise Exception(f"OpenAI記事生成エラー: {e}")

class AnthropicClient(BaseLLMClient):
    """Anthropic Claude APIクライアント"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model_name = "claude-3-haiku-20240307"  # コスト効率の良いモデル
        self._initialize()
    
    def _initialize(self):
        """Anthropic APIの初期化"""
        try:
            if self.api_key:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self._available = True
            else:
                self._available = False
        except ImportError:
            print("警告: anthropic パッケージがインストールされていません")
            self._available = False
        except Exception as e:
            print(f"Anthropic API初期化エラー: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        return self._available
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Anthropicのコスト計算"""
        input_cost_per_1k = 0.00025
        output_cost_per_1k = 0.00125
        return (input_tokens / 1000) * input_cost_per_1k + (output_tokens / 1000) * output_cost_per_1k
    
    def generate_article(self, prompt: str, temperature: float = 0.7) -> Tuple[str, int, int, float]:
        """Anthropic APIで記事を生成"""
        if not self.is_available():
            raise Exception("Anthropic APIが利用できません")
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            article = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            # コスト計算（Claude 3 Haiku）
            input_cost_per_1k = 0.00025  # $0.00025 per 1K input tokens
            output_cost_per_1k = 0.00125  # $0.00125 per 1K output tokens
            cost = (input_tokens / 1000) * input_cost_per_1k + (output_tokens / 1000) * output_cost_per_1k
            
            return article, input_tokens, output_tokens, cost
        except Exception as e:
            raise Exception(f"Anthropic記事生成エラー: {e}")

class LLMManager:
    """LLMプロバイダーを管理するクラス"""
    
    def __init__(self):
        self.clients = {
            "gemini": GeminiClient(),
            "openai": OpenAIClient(),
            "anthropic": AnthropicClient()
        }
        self.default_provider = "gemini"
    
    def get_available_providers(self) -> Dict[str, bool]:
        """利用可能なプロバイダーのリストを返す"""
        return {name: client.is_available() for name, client in self.clients.items()}
    
    def set_default_provider(self, provider: LLMProvider):
        """デフォルトプロバイダーを設定"""
        if provider in self.clients:
            self.default_provider = provider
        else:
            raise ValueError(f"未対応のプロバイダー: {provider}")
    
    def generate_article(self, prompt: str, temperature: float = 0.7, provider: Optional[LLMProvider] = None) -> Tuple[str, int, int, float, str]:
        """指定されたプロバイダーで記事を生成"""
        provider = provider or self.default_provider
        
        if provider not in self.clients:
            raise ValueError(f"未対応のプロバイダー: {provider}")
        
        client = self.clients[provider]
        if not client.is_available():
            # フォールバック: 利用可能な他のプロバイダーを試す
            for fallback_provider, fallback_client in self.clients.items():
                if fallback_client.is_available():
                    print(f"警告: {provider}が利用できないため、{fallback_provider}を使用します")
                    article, input_tokens, output_tokens, cost = fallback_client.generate_article(prompt, temperature)
                    return article, input_tokens, output_tokens, cost, fallback_provider
            
            raise Exception("利用可能なLLMプロバイダーがありません")
        
        article, input_tokens, output_tokens, cost = client.generate_article(prompt, temperature)
        return article, input_tokens, output_tokens, cost, provider

# グローバルインスタンス
llm_manager = LLMManager()

def get_llm_manager() -> LLMManager:
    """LLMマネージャーのインスタンスを取得"""
    return llm_manager
