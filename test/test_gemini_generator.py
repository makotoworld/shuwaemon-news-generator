def test_create_prompt():
    """プロンプト作成機能のテスト"""
    import pandas as pd
    from gemini_article_generator import create_prompt
    
    # テスト用データフレーム
    data = {
        'title': ['テスト記事'],
        'description': ['これはテスト用の記事です']
    }
    df = pd.DataFrame(data)
    
    prompt = create_prompt("AI", df)
    
    assert "AI" in prompt
    assert "テスト記事" in prompt
