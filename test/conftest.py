"""
pytest用の共通設定とフィクスチャ
"""
import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
import tempfile

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
                '日本のテクノロジー企業が世界市場で存在感'
            ],
            'description': [
                '宇宙開発が加速しています。民間企業も参入し、新たな時代が始まっています。',
                '人工知能技術の進化により、様々な分野で革新が起きている一方で、倫理的課題も浮上しています。',
                '環境問題に対する意識が高まり、持続可能な社会を目指す取り組みが世界中で増えています。',
                '日本発のテクノロジー企業が世界市場で躍進しており、特にAI分野で注目されています。'
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

# テスト用のAPI Keyモック
@pytest.fixture(scope="session", autouse=True)
def mock_api_key():
    """テスト用のダミーGoogle API Keyを設定"""
    original_api_key = os.environ.get("GOOGLE_API_KEY")
    
    # テスト用のダミーキーを設定（実際のAPIコールは行わない）
    if not original_api_key:
        os.environ["GOOGLE_API_KEY"] = "test_api_key_for_pytest"
    
    yield
    
    # テスト後に環境を元に戻す
    if not original_api_key:
        os.environ.pop("GOOGLE_API_KEY", None)
    else:
        os.environ["GOOGLE_API_KEY"] = original_api_key
