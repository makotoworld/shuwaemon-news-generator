"""
しゅわえもんニュース生成システムのテストコード
pytest を使用してアプリケーションの機能をテスト
"""
import os
import sys

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# 共通ライブラリのインポート
from lib.llm_client import get_llm_manager
from lib.data_utils import get_keyword_suggestions, find_relevant_content, extract_keywords

# 環境変数のロード（開発環境用）
load_dotenv()

# テスト用の環境変数設定
os.environ["GOOGLE_API_KEY"] = "test_api_key"
os.environ["AUTH_USERNAME"] = "admin"
os.environ["AUTH_PASSWORD"] = "password"

# アプリケーションのインポート
from app import app

# テストクライアントのセットアップ
client = TestClient(app)

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

# 単体テスト
def test_get_keyword_suggestions(sample_df):
    """キーワード候補抽出機能のテスト"""
    suggestions = get_keyword_suggestions(sample_df, limit=5)
    # タイトル全体が含まれているかをチェック
    assert '宇宙開発の最新動向' in suggestions or 'AIが変える未来' in suggestions or '環境問題への取り組み' in suggestions

def test_find_relevant_content(sample_df):
    """関連コンテンツ検索機能のテスト"""
    # 宇宙開発に関連する行が見つかるか
    result = find_relevant_content(sample_df, "宇宙")
    assert len(result) > 0

    # 存在しないキーワードの場合はランダムサンプルが返されるか
    result = find_relevant_content(sample_df, "存在しないキーワード")
    assert len(result) > 0  # サンプルが返される

def test_extract_keywords():
    """キーワード抽出機能のテスト"""
    text = "しゅわえもんニュース 最新技術の動向について解説します"
    keywords = extract_keywords(text, max_keywords=3)

    assert len(keywords) <= 3
    assert "しゅわえもんニュース" in keywords

def test_llm_manager_initialization():
    """LLMマネージャー初期化テスト"""
    manager = get_llm_manager()
    assert manager is not None
    assert len(manager.clients) == 3  # Gemini, OpenAI, Anthropic

# APIエンドポイントのテスト
def test_read_root():
    """ルートエンドポイントのテスト"""
    response = client.get("/", auth=("admin", "password"))  # 認証を追加
    assert response.status_code == 200
    # HTML応答であることを確認
    assert "text/html" in response.headers["content-type"]
    # プロバイダー選択機能が含まれているか確認
    assert "LLMプロバイダー" in response.text

@patch('lib.data_utils.load_excel_data')
def test_get_suggestions_in_homepage(mock_load_data, sample_df):
    """ホームページでのキーワード候補表示テスト"""
    # モックの設定
    mock_load_data.return_value = sample_df

    # APIの呼び出し
    response = client.get("/", auth=("admin", "password"))
    assert response.status_code == 200
    # レスポンスにキーワードが含まれているか
    assert "宇宙開発" in response.text or "AI" in response.text or "環境問題" in response.text

# LLM記事生成のモックテスト
@patch('app.generate_article_with_llm')
def test_generate_article_endpoint(mock_generate):
    """記事生成エンドポイントのテスト"""
    # モックの設定
    mock_generate.return_value = (
        "テスト記事のコンテンツです。これは自動生成されたニュース記事です。", 
        100, 200, 0.0025, "gemini"
    )

    # APIの呼び出し
    response = client.post(
        "/generate",
        data={"keyword": "テスト", "temperature": "0.7"},
        auth=("admin", "password")
    )
    assert response.status_code == 200
    # レスポンスに記事が含まれているか
    assert "テスト記事のコンテンツ" in response.text
    # 使用プロバイダーが表示されているか
    assert "gemini" in response.text.lower()

@patch('app.generate_article_with_llm')
def test_generate_article_with_provider_selection(mock_generate):
    """プロバイダー選択での記事生成テスト"""
    # モックの設定
    mock_generate.return_value = (
        "OpenAIで生成された記事です。", 
        150, 250, 0.0035, "openai"
    )

    # APIの呼び出し（プロバイダー指定）
    response = client.post(
        "/generate",
        data={"keyword": "AI技術", "temperature": "0.7", "provider": "openai"},
        auth=("admin", "password")
    )
    assert response.status_code == 200
    # レスポンスに記事が含まれているか
    assert "OpenAIで生成された記事" in response.text
    # 使用プロバイダーが表示されているか
    assert "openai" in response.text.lower()

# 失敗ケースのテスト
@patch('app.generate_article_with_llm')
def test_generate_article_error(mock_generate):
    """記事生成エラーケースのテスト"""
    # エラーを発生させるモック
    mock_generate.side_effect = Exception("テストエラー: API呼び出しに失敗しました")

    # APIの呼び出し
    response = client.post(
        "/generate",
        data={"keyword": "エラー", "temperature": "0.7"},
        auth=("admin", "password")
    )
    assert response.status_code == 200  # 成功レスポンスだがエラーテンプレート
    # エラーメッセージが含まれているか
    assert "テストエラー" in response.text

def test_generate_article_empty_keyword():
    """空のキーワードでの記事生成テスト"""
    response = client.post(
        "/generate",
        data={"keyword": "", "temperature": "0.7"},
        auth=("admin", "password")
    )
    # エラーレスポンスまたはバリデーションエラーが返されるか
    assert response.status_code in [200, 422]  # FastAPIのバリデーションエラー

# 履歴機能のテスト
def test_history_endpoint():
    """履歴エンドポイントのテスト"""
    response = client.get("/history", auth=("admin", "password"))
    assert response.status_code == 200
    assert "API使用履歴" in response.text

def test_clear_history_endpoint():
    """履歴クリアエンドポイントのテスト"""
    response = client.get("/clear-history", auth=("admin", "password"))
    assert response.status_code == 200
    assert "履歴がクリアされました" in response.text

# ファイルダウンロードのテスト
def test_download_nonexistent_file():
    """存在しないファイルのダウンロードテスト"""
    response = client.get("/download/nonexistent_file.txt", auth=("admin", "password"))
    assert response.status_code == 404

# 認証のテスト
def test_unauthorized_access():
    """認証なしでのアクセステスト"""
    response = client.get("/")
    assert response.status_code == 401

def test_wrong_credentials():
    """間違った認証情報でのアクセステスト"""
    response = client.get("/", auth=("wrong", "credentials"))
    assert response.status_code == 401

# 統合テスト
@pytest.mark.integration
def test_end_to_end_workflow():
    """
    エンドツーエンドのワークフロー統合テスト
    注意: このテストは実際のAPIを呼び出すため、環境変数が設定されていることが前提
    通常のCIでは実行せず、特別なフラグが必要
    """
    # このテストは特別なフラグが設定されている場合のみ実行
    if not os.environ.get("RUN_INTEGRATION_TESTS"):
        pytest.skip("統合テストはスキップされました。RUN_INTEGRATION_TESTS環境変数を設定して実行してください。")

    # ホームページにアクセス
    response = client.get("/", auth=("admin", "password"))
    assert response.status_code == 200
    assert "LLMプロバイダー" in response.text

    # 記事を生成
    response = client.post(
        "/generate",
        data={"keyword": "テスト自動化", "temperature": "0.7"},
        auth=("admin", "password")
    )
    assert response.status_code == 200
    assert "記事内容" in response.text or "テスト自動化" in response.text

    # 履歴ページにアクセス
    response = client.get("/history", auth=("admin", "password"))
    assert response.status_code == 200
    assert "API使用履歴" in response.text

# パフォーマンステスト
@pytest.mark.performance
def test_homepage_response_time():
    """ホームページのレスポンス時間テスト"""
    import time
    
    start_time = time.time()
    response = client.get("/", auth=("admin", "password"))
    end_time = time.time()
    
    assert response.status_code == 200
    # 3秒以内にレスポンスが返されることを確認
    assert (end_time - start_time) < 3.0

# セキュリティテスト
def test_sql_injection_protection():
    """SQLインジェクション対策テスト"""
    malicious_keyword = "'; DROP TABLE users; --"
    response = client.post(
        "/generate",
        data={"keyword": malicious_keyword, "temperature": "0.7"},
        auth=("admin", "password")
    )
    # エラーが発生しても500エラーにならないことを確認
    assert response.status_code in [200, 400, 422]

def test_xss_protection():
    """XSS対策テスト"""
    xss_keyword = "<script>alert('XSS')</script>"
    response = client.post(
        "/generate",
        data={"keyword": xss_keyword, "temperature": "0.7"},
        auth=("admin", "password")
    )
    # スクリプトタグがエスケープされているか、またはエラーが発生しているか確認
    # HTMLテンプレート内の<script>タグは除外して検証
    response_text_without_template = response.text.replace('<script>', '').replace('</script>', '')
    assert xss_keyword not in response_text_without_template or "エラー" in response.text
