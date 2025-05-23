"""
Gemini API関連の共通コード
"""
import os
import google.generativeai as genai
from typing import Tuple, Optional, Dict, Any

# Gemini API設定
MODEL_NAME = "gemini-2.0-flash-lite"  # 最新の高速モデル

def initialize_genai(api_key: Optional[str] = None) -> bool:
    """Gemini APIの初期化"""
    try:
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not api_key:
            print("警告: GOOGLE_API_KEY環境変数が設定されていません")
            return False
            
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"Gemini API初期化エラー: {e}")
        return False

def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Gemini APIの使用コストを計算
    Gemini 2.0 Flash Liteの料金（2025年5月現在）
    """
    input_cost_per_1k = 0.0003  # $0.0003 per 1K input tokens
    output_cost_per_1k = 0.0006  # $0.0006 per 1K output tokens
    
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    
    return input_cost + output_cost

def generate_article(prompt: str, temperature: float = 0.7) -> Tuple[str, int, int, float]:
    """
    Gemini APIを使用して記事を生成し、トークン数とコストを返す
    
    Returns:
        Tuple[str, int, int, float]: 生成された記事、入力トークン数、出力トークン数、コスト
    """
    try:
        # モデルの設定
        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
        
        # モデルの作成と記事生成
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt, generation_config=generation_config)
        
        # トークン数の概算（実際のAPIレスポンスによって異なる可能性あり）
        input_tokens = len(prompt.split())
        output_tokens = len(response.text.split())
        
        # コスト計算
        cost = calculate_cost(input_tokens, output_tokens)
        
        return response.text, input_tokens, output_tokens, cost
    except Exception as e:
        print(f"記事生成エラー: {e}")
        raise
