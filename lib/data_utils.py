"""
データ処理の共通コード
"""
import os
import pandas as pd
from typing import List, Dict, Any, Optional

def load_excel_data(file_path: str) -> pd.DataFrame:
    """データファイルからExcelデータを読み込む"""
    try:
        if os.path.exists(file_path):
            return pd.read_excel(file_path)
        else:
            print(f"警告: データファイル '{file_path}' が見つかりません")
            return pd.DataFrame()
    except Exception as e:
        print(f"Excelファイルの読み込みエラー: {e}")
        return pd.DataFrame()

def get_keyword_suggestions(df: pd.DataFrame, limit: int = 10) -> List[str]:
    """
    データフレームからキーワード候補を抽出
    """
    if df.empty:
        return []
    
    # タイトルから単語を抽出
    all_words = []
    for title in df['title'].fillna(''):
        if not isinstance(title, str):
            continue
        # 最低2文字以上の単語を抽出
        words = [word for word in title.split() if len(word) >= 2]
        all_words.extend(words)
    
    # 出現頻度でソート
    word_counts = pd.Series(all_words).value_counts()
    return word_counts.index.tolist()[:limit]

def find_relevant_content(df: pd.DataFrame, keyword: str, num_samples: int = 5) -> pd.DataFrame:
    """キーワードに関連する内容をデータフレームから検索"""
    if df.empty:
        return pd.DataFrame()
        
    # NaN値を空文字列に置換
    df_clean = df.fillna('')
    
    # キーワードを含む行を抽出
    keyword_lower = keyword.lower()
    mask = (
        df_clean['title'].str.lower().str.contains(keyword_lower) | 
        df_clean['description'].str.lower().str.contains(keyword_lower)
    )
    relevant_df = df_clean[mask]
    
    if len(relevant_df) == 0:
        # 関連コンテンツがない場合はランダムサンプル
        return df_clean.sample(min(num_samples, len(df_clean)))
    
    # 関連コンテンツのサンプルを返す
    return relevant_df.sample(min(num_samples, len(relevant_df)))

def create_prompt(keyword: str, examples_df: pd.DataFrame) -> str:
    """Gemini APIへのプロンプトを作成（信憑性情報付き）"""
    # 例を整形
    examples_text = ""
    for i, row in examples_df.iterrows():
        examples_text += f"タイトル: {row['title']}\n"
        examples_text += f"概要: {row['description']}\n\n"
    
    # プロンプトテンプレート
    prompt = f"""
あなたは「しゅわえもんニュース」というYouTubeチャンネルの記事を書く専門ライターです。
以下の特徴を持つニュース記事を日本語で作成してください:

1. キーワード「{keyword}」に関する最新のニュース記事
2. しゅわえもんニュースのスタイルで書かれた
3. 事実ベースで読者が理解しやすい文章
4. 見出し、導入部、本文、結論、信憑性情報という構成

以下はしゅわえもんニュースの例です:

{examples_text}

これらの例を参考にして、「{keyword}」についての記事を作成してください。

**重要**: 記事の信憑性を高めるため、以下の情報も含めてください:
- 情報源の種類（政府機関、学術機関、専門機関など）
- 関連する専門分野や研究領域
- 事実確認のポイント
- 参考になる検索キーワード

記事のフォーマットは以下の通りです:

[タイトル]

[導入部 - キーワードについての簡単な紹介と記事の概要]

[本文 - キーワードに関する詳細な内容]

[結論 - 要点のまとめと今後の展望]

[信憑性情報]
- 情報源: [この記事の内容に関連する信頼できる情報源の種類]
- 専門分野: [関連する学術・専門分野]
- 確認ポイント: [読者が事実確認する際のポイント]
- 検索キーワード: [さらに詳しく調べるための検索キーワード]
- 注意事項: [情報の解釈や利用時の注意点]
"""
    return prompt

def extract_keywords(text, max_keywords=5):
    """簡易的なキーワード抽出（実際にはより高度な自然言語処理を使用することを推奨）"""
    # この実装は非常に簡易的です。実際の使用では、MeCabやJanomeなどの日本語形態素解析器を
    # 使用することを推奨します。
    
    # 記号や一般的な助詞・助動詞などを除外
    stop_words = ['の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'も', 'な', 'です', 'ます']
    
    # 単語に分割して頻度をカウント（実際には形態素解析を使用すべき）
    words = text.replace('\n', ' ').split(' ')
    word_count = {}
    
    for word in words:
        if len(word) > 1 and word not in stop_words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    
    # 出現頻度順にソート
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    
    # 上位のキーワードを返す（重複を除く）
    keywords = []
    for word, _ in sorted_words:
        if word not in keywords:
            keywords.append(word)
            if len(keywords) >= max_keywords:
                break
    
    return keywords
