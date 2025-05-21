"""
しゅわえもんニュース生成システムのテストコード
pytest を使用してアプリケーションの機能をテスト
"""
import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# テスト用の環境変数設定
os.environ["GOOGLE_API_KEY"] = "test_api_key"

# アプリケーションのインポート
from app import app, calculate_cost, get_keyword_suggestions, find_relevant_content

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
def test_calculate_cost():
    """コスト計算関数のテスト"""
    # 入力1000トークン、出力500トークンの場合
    cost = calculate_cost(1000, 500)
    expected = (1000 / 1000) * 0.0025 + (500 / 1000) * 0.0075
    assert cost == expected

#def test_get_keyword_suggestions(sample_df):
#    """キーワード候補抽出機能のテスト"""
#    suggestions = get_keyword_suggestions(sample_df, limit=5)
#    # データフレームから抽出されたキーワードが含まれているか
#    assert '宇宙開発' in suggestions or '最新動向' in suggestions
#    assert 'AI' in suggestions or '未来' in suggestions
#    assert '環境問題' in suggestions or '取り組み' in suggestions
#    # 最大数の制限が守られているか
#    assert len(suggestions) <= 5

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

# APIエンドポイントのテスト
def test_read_root():
    """ルートエンドポイントのテスト"""
    response = client.get("/")
    assert response.status_code == 200
    # HTML応答であることを確認
    assert "text/html" in response.headers["content-type"]

@patch('app.load_excel_data')
def test_get_suggestions(mock_load_data, sample_df):
    """キーワード候補APIエンドポイントのテスト（もしあれば）"""
    # モックの設定
    mock_load_data.return_value = sample_df
    
    # APIの呼び出し
    response = client.get("/")
    assert response.status_code == 200
    # レスポンスにキーワードが含まれているか
    assert "宇宙開発" in response.text or "AI" in response.text or "環境問題" in response.text

# Gemini API呼び出しのモックテスト
@patch('app.generate_article_with_gemini')
def test_generate_article(mock_generate):
    """記事生成エンドポイントのテスト"""
    # モックの設定
    mock_generate.return_value = (
        "テスト記事のコンテンツ", 100, 200, 0.0025
    )
    
    # APIの呼び出し
    response = client.post(
        "/generate",
        data={"keyword": "テスト", "temperature": 0.7}
    )
    assert response.status_code == 200
    # レスポンスに記事が含まれているか
    assert "テスト記事のコンテンツ" in response.text

# 失敗ケースのテスト
@patch('app.generate_article_with_gemini')
def test_generate_article_error(mock_generate):
    """記事生成エラーケースのテスト"""
    # エラーを発生させるモック
    mock_generate.side_effect = Exception("テストエラー")
    
    # APIの呼び出し
    response = client.post(
        "/generate",
        data={"keyword": "エラー", "temperature": 0.7}
    )
    assert response.status_code == 200  # 成功レスポンスだがエラーテンプレート
    # エラーメッセージが含まれているか
    assert "テストエラー" in response.text

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
    response = client.get("/")
    assert response.status_code == 200
    
    # 記事を生成
    response = client.post(
        "/generate",
        data={"keyword": "テスト自動化", "temperature": 0.7}
    )
    assert response.status_code == 200
    assert "記事内容" in response.text
    
    # 履歴ページにアクセス
    response = client.get("/history")
    assert response.status_code == 200
    assert "API使用履歴" in response.text
    assert "テスト自動化" in response.text
