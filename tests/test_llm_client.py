"""
LLMクライアントシステムのテストコード
複数のLLMプロバイダーに対応したテスト
"""
import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# 共通ライブラリのインポート
from lib.llm_client import (
    LLMManager, 
    GeminiClient, 
    OpenAIClient, 
    AnthropicClient,
    get_llm_manager
)
from lib.data_utils import create_prompt

# 環境変数のロード（開発環境用）
load_dotenv()

# テスト用のダミーデータ
@pytest.fixture
def sample_df():
    """テスト用のサンプルデータフレーム"""
    return pd.DataFrame({
        'title': ['宇宙開発の最新動向', 'AIが変える未来', '環境問題への取り組み'],
        'description': [
            '宇宙開発が加速しています。民間企業も参入し、新たな時代に。',
            '人工知能技術の進化により、様々な分野で革新が起きています。',
            '環境問題に対する意識が高まり、持続可能な取り組みが増えています。'
        ]
    })

@pytest.fixture
def sample_prompt():
    """テスト用のサンプルプロンプト"""
    return """
    以下の情報を参考に、「AI技術」に関するニュース記事を生成してください。
    
    参考情報:
    - AIが変える未来
    - 人工知能技術の進化により、様々な分野で革新が起きています。
    
    記事の要件:
    - 800文字程度
    - 客観的で正確な情報
    - 読みやすい構成
    """

# LLMクライアント個別テスト
class TestGeminiClient:
    """Geminiクライアントのテスト"""
    
    def test_initialization_with_api_key(self):
        """APIキーありでの初期化テスト"""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            client = GeminiClient()
            assert client.is_available() == True
    
    def test_initialization_without_api_key(self):
        """APIキーなしでの初期化テスト"""
        with patch.dict(os.environ, {}, clear=True):
            client = GeminiClient()
            assert client.is_available() == False
    
    def test_calculate_cost(self):
        """コスト計算テスト"""
        client = GeminiClient()
        cost = client.calculate_cost(1000, 500)
        # gemini-2.0-flash-liteの料金: 入力$0.0003/1K, 出力$0.0006/1K
        expected = (1000 / 1000) * 0.0003 + (500 / 1000) * 0.0006
        assert cost == expected
    
    @patch('google.generativeai.GenerativeModel')
    def test_generate_article_success(self, mock_model, sample_prompt):
        """記事生成成功テスト"""
        # モックの設定
        mock_response = MagicMock()
        mock_response.text = "生成されたテスト記事の内容です。"
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 200
        
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_instance
        
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            client = GeminiClient()
            article, input_tokens, output_tokens, cost = client.generate_article(sample_prompt, 0.7)

            assert article == "生成されたテスト記事の内容です。"
            # トークン数は概算値なので、実際の値と一致しない場合がある
            assert input_tokens > 0
            assert output_tokens > 0
            assert cost > 0

class TestOpenAIClient:
    """OpenAIクライアントのテスト"""
    
    def test_initialization_with_api_key(self):
        """APIキーありでの初期化テスト"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            client = OpenAIClient()
            assert client.is_available() == True
    
    def test_initialization_without_api_key(self):
        """APIキーなしでの初期化テスト"""
        with patch.dict(os.environ, {}, clear=True):
            client = OpenAIClient()
            assert client.is_available() == False
    
    def test_calculate_cost(self):
        """コスト計算テスト"""
        client = OpenAIClient()
        cost = client.calculate_cost(1000, 500)
        # gpt-4o-miniの料金: 入力$0.00015/1K, 出力$0.0006/1K
        expected = (1000 / 1000) * 0.00015 + (500 / 1000) * 0.0006
        assert cost == expected
    
    @patch('openai.OpenAI')
    def test_generate_article_success(self, mock_openai, sample_prompt):
        """記事生成成功テスト"""
        # モックの設定
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "OpenAIで生成されたテスト記事です。"
        mock_response.usage.prompt_tokens = 150
        mock_response.usage.completion_tokens = 250
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            client = OpenAIClient()
            article, input_tokens, output_tokens, cost = client.generate_article(sample_prompt, 0.7)
            
            assert article == "OpenAIで生成されたテスト記事です。"
            assert input_tokens == 150
            assert output_tokens == 250
            assert cost > 0

class TestAnthropicClient:
    """Anthropicクライアントのテスト"""
    
    def test_initialization_with_api_key(self):
        """APIキーありでの初期化テスト"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            client = AnthropicClient()
            assert client.is_available() == True
    
    def test_initialization_without_api_key(self):
        """APIキーなしでの初期化テスト"""
        with patch.dict(os.environ, {}, clear=True):
            client = AnthropicClient()
            assert client.is_available() == False
    
    def test_calculate_cost(self):
        """コスト計算テスト"""
        client = AnthropicClient()
        cost = client.calculate_cost(1000, 500)
        # claude-3-haikuの料金: 入力$0.00025/1K, 出力$0.00125/1K
        expected = (1000 / 1000) * 0.00025 + (500 / 1000) * 0.00125
        assert cost == expected
    
    @patch('anthropic.Anthropic')
    def test_generate_article_success(self, mock_anthropic, sample_prompt):
        """記事生成成功テスト"""
        # モックの設定
        mock_response = MagicMock()
        mock_response.content[0].text = "Claudeで生成されたテスト記事です。"
        mock_response.usage.input_tokens = 120
        mock_response.usage.output_tokens = 180
        
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            client = AnthropicClient()
            article, input_tokens, output_tokens, cost = client.generate_article(sample_prompt, 0.7)
            
            assert article == "Claudeで生成されたテスト記事です。"
            assert input_tokens == 120
            assert output_tokens == 180
            assert cost > 0

# LLMマネージャーのテスト
class TestLLMManager:
    """LLMマネージャーのテスト"""
    
    def test_initialization(self):
        """初期化テスト"""
        manager = LLMManager()
        assert len(manager.clients) == 3  # Gemini, OpenAI, Anthropic
    
    def test_get_available_providers(self):
        """利用可能プロバイダー取得テスト"""
        with patch.dict(os.environ, {
            'GOOGLE_API_KEY': 'test_key',
            'OPENAI_API_KEY': 'test_key'
        }, clear=True):
            manager = LLMManager()
            available = manager.get_available_providers()
            
            assert available['gemini'] == True
            assert available['openai'] == True
            assert available['anthropic'] == False
    
    @patch('lib.llm_client.GeminiClient.generate_article')
    def test_generate_article_with_specific_provider(self, mock_generate, sample_prompt):
        """特定プロバイダー指定での記事生成テスト"""
        mock_generate.return_value = ("テスト記事", 100, 200, 0.05)
        
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            manager = LLMManager()
            article, input_tokens, output_tokens, cost, used_provider = manager.generate_article(
                sample_prompt, 0.7, "gemini"
            )
            
            assert article == "テスト記事"
            assert used_provider == "gemini"
            mock_generate.assert_called_once()
    
    @patch('lib.llm_client.GeminiClient.generate_article')
    @patch('lib.llm_client.OpenAIClient.generate_article')
    def test_generate_article_auto_selection(self, mock_openai, mock_gemini, sample_prompt):
        """自動選択での記事生成テスト"""
        # Geminiが利用不可、OpenAIが利用可能な場合
        mock_gemini.side_effect = Exception("API Key not available")
        mock_openai.return_value = ("OpenAI記事", 150, 250, 0.08)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}, clear=True):
            manager = LLMManager()
            article, input_tokens, output_tokens, cost, used_provider = manager.generate_article(
                sample_prompt, 0.7
            )
            
            assert article == "OpenAI記事"
            assert used_provider == "openai"

# プロンプト作成のテスト
def test_create_prompt(sample_df):
    """プロンプト作成機能のテスト"""
    relevant_content = sample_df.head(2)  # 最初の2行を使用
    prompt = create_prompt("AI技術", relevant_content)
    
    assert "AI技術" in prompt
    assert "宇宙開発の最新動向" in prompt or "AIが変える未来" in prompt
    assert "信憑性情報" in prompt  # 新しく追加された信憑性セクション

# 統合テスト
def test_get_llm_manager():
    """LLMマネージャー取得関数のテスト"""
    manager = get_llm_manager()
    assert isinstance(manager, LLMManager)
    assert len(manager.clients) == 3

# エラーハンドリングテスト
class TestErrorHandling:
    """エラーハンドリングのテスト"""
    
    @patch('lib.llm_client.GeminiClient.generate_article')
    def test_api_error_handling(self, mock_generate, sample_prompt):
        """API エラーハンドリングテスト"""
        mock_generate.side_effect = Exception("API rate limit exceeded")
        
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            manager = LLMManager()
            
            with pytest.raises(Exception):
                manager.generate_article(sample_prompt, 0.7, "gemini")
    
    def test_no_available_providers(self, sample_prompt):
        """利用可能プロバイダーなしのテスト"""
        with patch.dict(os.environ, {}, clear=True):
            manager = LLMManager()
            
            with pytest.raises(Exception) as exc_info:
                manager.generate_article(sample_prompt, 0.7)
            
            assert "利用可能なLLMプロバイダーがありません" in str(exc_info.value)
