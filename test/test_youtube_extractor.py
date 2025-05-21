def test_extract_keywords():
    """キーワード抽出機能のテスト"""
    from youtube_data_extractor import extract_keywords
    
    text = "しゅわえもんニュース 最新技術の動向について解説します"
    keywords = extract_keywords(text, max_keywords=3)
    
    assert len(keywords) <= 3
    assert "しゅわえもんニュース" in keywords
