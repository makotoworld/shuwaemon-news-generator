"""
pytest用の共通設定とフィクスチャ
新しいLLMクライアントシステムに対応
"""
import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
import tempfile
from unittest.mock import patch, MagicMock

# テスト中に生成される記事ファイルを格納するための一時ディレクトリ
@pytest.fixture(scope="session")
def temp_output_dir():
    """テスト用の一時出力ディレクトリを作成"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        original_output_dir = os.environ.get("OUTPUT_DIR")
        os.environ["OUTPUT_DIR"] = tmpdirname
        yield tmpdirname
        # 環境を元に戻す
        if original_output_dir:
            os.environ["OUTPUT_DIR"] = original_output_dir
        else:
            os.environ.pop("OUTPUT_DIR", None)

# テスト用のダミーExcelデータ
@pytest.fixture(scope="session")
def sample_excel_file():
    """テスト用のサンプルExcelファイルを作成"""
    # 一時ファイルの作成
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        # サンプルデータ
        df = pd.DataFrame({
            'title': [
                '宇宙開発の最新動向が明らかに', 
                'AI技術が進化し続ける中での課題',
                '環境問題に対する新たな取り組み',
                '日本のテクノロジー企業が世界市場で存在感',
                'しゅわえもんニュースが伝える最新情報',
                '量子コンピューターの実用化が加速',
                'バイオテクノロジーの革新的進歩',
                'サステナブルエネルギーの未来展望'
            ],
            'description': [
                '宇宙開発が加速しています。民間企業も参入し、新たな時代が始まっています。',
                '人工知能技術の進化により、様々な分野で革新が起きている一方で、倫理的課題も浮上しています。',
                '環境問題に対する意識が高まり、持続可能な社会を目指す取り組みが世界中で増えています。',
                '日本発のテクノロジー企業が世界市場で躍進しており、特にAI分野で注目されています。',
                'しゅわえもんニュースでは、最新の技術動向や社会情勢について詳しく解説しています。',
                '量子コンピューターの技術が実用段階に入り、従来のコンピューターでは不可能だった計算が可能になります。',
                'バイオテクノロジー分野では、遺伝子編集技術の進歩により医療分野での応用が期待されています。',
                '再生可能エネルギーの技術革新により、持続可能なエネルギー社会の実現が近づいています。'
            ]
        })
        
        # Excelファイルとして保存
        df.to_excel(tmp.name, index=False)
        excel_path = tmp.name
    
    # テスト実行中に環境変数を設定
    original_data_file = os.environ.get("DATA_FILE")
    os.environ["DATA_FILE"] = excel_path
    
    yield excel_path
    
    # テスト後に環境を元に戻し、一時ファイルを削除
    if original_data_file:
        os.environ["DATA_FILE"] = original_data_file
    else:
        os.environ.pop("DATA_FILE", None)
    
    os.unlink(excel_path)

# テスト用のAPI Keyモック（複数プロバイダー対応）
@pytest.fixture(scope="session", autouse=True)
def mock_api_keys():
    """テスト用のダミーAPI Keyを設定（全プロバイダー対応）"""
    original_keys = {
        'GOOGLE_API_KEY': os.environ.get("GOOGLE_API_KEY"),
        'OPENAI_API_KEY': os.environ.get("OPENAI_API_KEY"),
        'ANTHROPIC_API_KEY': os.environ.get("ANTHROPIC_API_KEY"),
        'AUTH_USERNAME': os.environ.get("AUTH_USERNAME"),
        'AUTH_PASSWORD': os.environ.get("AUTH_PASSWORD")
    }
    
    # テスト用のダミーキーを設定
    test_keys = {
        'GOOGLE_API_KEY': 'test_google_api_key_for_pytest',
        'OPENAI_API_KEY': 'test_openai_api_key_for_pytest',
        'ANTHROPIC_API_KEY': 'test_anthropic_api_key_for_pytest',
        'AUTH_USERNAME': 'admin',
        'AUTH_PASSWORD': 'password'
    }
    
    for key, value in test_keys.items():
        if not original_keys[key]:
            os.environ[key] = value
    
    yield
    
    # テスト後に環境を元に戻す
    for key, original_value in original_keys.items():
        if not original_value:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

# LLMクライアントのモック用フィクスチャ
@pytest.fixture
def mock_llm_responses():
    """LLMクライアントのレスポンスをモック化"""
    responses = {
        'gemini': {
            'article': 'Geminiで生成されたテスト記事です。最新の技術動向について詳しく解説します。',
            'input_tokens': 100,
            'output_tokens': 200,
            'cost': 0.0009
        },
        'openai': {
            'article': 'OpenAIで生成されたテスト記事です。AIの進歩について分析します。',
            'input_tokens': 150,
            'output_tokens': 250,
            'cost': 0.00075
        },
        'anthropic': {
            'article': 'Claudeで生成されたテスト記事です。倫理的な観点から技術を考察します。',
            'input_tokens': 120,
            'output_tokens': 180,
            'cost': 0.000525
        }
    }
    return responses

# テスト用のサンプルデータフレーム
@pytest.fixture
def sample_dataframe():
    """テスト用の標準的なサンプルデータフレーム"""
    return pd.DataFrame({
        'title': [
            '宇宙開発の最新動向',
            'AIが変える未来',
            '環境問題への取り組み',
            'しゅわえもんニュースの特集'
        ],
        'description': [
            '宇宙開発が加速しています。民間企業も参入し、新たな時代に。',
            '人工知能技術の進化により、様々な分野で革新が起きています。',
            '環境問題に対する意識が高まり、持続可能な取り組みが増えています。',
            'しゅわえもんニュースでは最新の技術動向を詳しく解説しています。'
        ]
    })

# テスト用のプロンプトサンプル
@pytest.fixture
def sample_prompt():
    """テスト用の標準的なプロンプト"""
    return """以下の情報を参考に、「AI技術」に関するニュース記事を生成してください。

参考情報:
- AIが変える未来
- 人工知能技術の進化により、様々な分野で革新が起きています。

記事の要件:
- 800文字程度
- 客観的で正確な情報
- 読みやすい構成
- しゅわえもんニュースらしい視点

信憑性情報:
- 情報源の種類: 技術レポート、学術論文
- 関連する専門分野: 人工知能、機械学習
- 事実確認のポイント: 技術の実用性、導入事例
- 参考検索キーワード: AI技術動向、機械学習応用
- 注意事項: 技術の限界や課題も含めて報告"""

# パフォーマンステスト用の設定
@pytest.fixture
def performance_config():
    """パフォーマンステスト用の設定"""
    return {
        'max_response_time': 3.0,  # 最大レスポンス時間（秒）
        'max_memory_usage': 100,   # 最大メモリ使用量（MB）
        'concurrent_requests': 5   # 同時リクエスト数
    }

# テストマーカーの設定
def pytest_configure(config):
    """pytestの設定"""
    config.addinivalue_line(
        "markers", "integration: 統合テスト（実際のAPIを使用）"
    )
    config.addinivalue_line(
        "markers", "performance: パフォーマンステスト"
    )
    config.addinivalue_line(
        "markers", "security: セキュリティテスト"
    )
    config.addinivalue_line(
        "markers", "slow: 実行時間の長いテスト"
    )

# テスト実行前の共通セットアップ
@pytest.fixture(autouse=True)
def setup_test_environment():
    """各テスト実行前の共通セットアップ"""
    # テスト用の一時ディレクトリを作成
    os.makedirs("generated_articles", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    yield
    
    # テスト後のクリーンアップは必要に応じて実装

# モック用のヘルパー関数
@pytest.fixture
def mock_llm_manager():
    """LLMマネージャーのモック"""
    with patch('lib.llm_client.LLMManager') as mock_manager_class:
        mock_manager = MagicMock()
        mock_manager.get_available_providers.return_value = {
            'gemini': True,
            'openai': True,
            'anthropic': False
        }
        mock_manager.generate_article.return_value = (
            "モックで生成されたテスト記事です。",
            100, 200, 0.0025, "gemini"
        )
        mock_manager_class.return_value = mock_manager
        yield mock_manager
