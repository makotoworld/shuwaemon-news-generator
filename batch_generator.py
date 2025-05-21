"""
複数キーワードの記事を一括生成するスクリプト
"""
import pandas as pd
import google.generativeai as genai
import os
import time
import argparse
from datetime import datetime

# Gemini APIからimportされた基本機能を使用
from gemini_article_generator import (
    initialize_genai, load_excel_data, find_relevant_content,
    create_prompt, generate_article, save_to_file
)

def extract_keywords_from_excel(df, column_name='keywords', separator=',', min_length=2, max_keywords=None):
    """
    Excelファイルから抽出したキーワードのリストを作成
    
    Parameters:
    - df: データフレーム
    - column_name: キーワードが含まれる列名
    - separator: キーワードの区切り文字
    - min_length: 最小文字数（これ未満は除外）
    - max_keywords: 最大キーワード数（Noneの場合は無制限）
    
    Returns:
    - キーワードのリスト
    """
    # キーワード列が存在するか確認
    if column_name not in df.columns:
        print(f"警告: '{column_name}' 列がExcelファイルに存在しません。")
        # タイトルから単語を抽出
        all_words = []
        for title in df['title'].fillna(''):
            words = [word for word in title.split() if len(word) >= min_length]
            all_words.extend(words)
        
        # 出現頻度でソート
        word_counts = pd.Series(all_words).value_counts()
        keywords = word_counts.index.tolist()
    else:
        # キーワード列から抽出
        all_keywords = []
        for kw_string in df[column_name].fillna(''):
            if not isinstance(kw_string, str):
                continue
                
            keywords_list = [kw.strip() for kw in kw_string.split(separator) if len(kw.strip()) >= min_length]
            all_keywords.extend(keywords_list)
        
        # 重複を削除
        keywords = list(set(all_keywords))
    
    # 最大キーワード数で制限
    if max_keywords is not None:
        keywords = keywords[:max_keywords]
    
    return keywords

def create_output_directory(dir_name="generated_articles"):
    """出力ディレクトリを作成"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def save_article_to_directory(article, keyword, directory):
    """生成された記事を指定ディレクトリに保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{keyword.replace(' ', '_')}_{timestamp}.txt"
    filepath = os.path.join(directory, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(article)
        return filepath
    except Exception as e:
        print(f"ファイル保存エラー: {e}")
        return None

def generate_articles_batch(keywords, excel_file, output_dir, temperature=0.7, delay=2):
    """
    キーワードリストに基づいてバッチで記事を生成
    
    Parameters:
    - keywords: 記事生成に使用するキーワードのリスト
    - excel_file: Excelデータファイルのパス
    - output_dir: 生成した記事を保存するディレクトリ
    - temperature: 生成の多様性（0.0-1.0）
    - delay: API呼び出し間の遅延（秒）
    """
    # Excelデータの読み込み
    df = load_excel_data(excel_file)
    if df is None:
        return
    
    # 出力ディレクトリの作成
    directory = create_output_directory(output_dir)
    
    # 結果のサマリー
    results = {
        "success": 0,
        "failed": 0,
        "files": []
    }
    
    # 各キーワードに対して記事を生成
    for i, keyword in enumerate(keywords):
        print(f"\n--- キーワード {i+1}/{len(keywords)}: '{keyword}' の記事を生成中 ---")
        
        try:
            # 関連コンテンツの抽出
            relevant_content = find_relevant_content(df, keyword)
            
            # プロンプトの作成
            prompt = create_prompt(keyword, relevant_content)
            
            # 記事の生成
            start_time = time.time()
            article = generate_article(prompt, temperature)
            end_time = time.time()
            
            print(f"記事生成完了（処理時間: {end_time - start_time:.2f}秒）")
            
            # ファイルに保存
            filepath = save_article_to_directory(article, keyword, directory)
            
            if filepath:
                print(f"記事を保存しました: {filepath}")
                results["success"] += 1
                results["files"].append(filepath)
            else:
                print(f"記事の保存に失敗しました: {keyword}")
                results["failed"] += 1
        
        except Exception as e:
            print(f"キーワード '{keyword}' の記事生成中にエラーが発生しました: {e}")
            results["failed"] += 1
        
        # API制限を避けるための遅延
        if i < len(keywords) - 1:
            print(f"{delay}秒間待機中...")
            time.sleep(delay)
    
    return results

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="GoogleのGemini APIを使用した一括記事生成ツール")
    parser.add_argument("--file", default="shuwaimon_news_data.xlsx", help="Excel データファイルのパス")
    parser.add_argument("--output", default="generated_articles", help="出力ディレクトリ")
    parser.add_argument("--temp", type=float, default=0.7, help="生成の多様性（0.0-1.0）")
    parser.add_argument("--delay", type=int, default=2, help="API呼び出し間の遅延（秒）")
    parser.add_argument("--max", type=int, default=10, help="生成する最大記事数")
    parser.add_argument("--keywords", help="カンマ区切りのキーワードリスト（指定しない場合はExcelから抽出）")
    args = parser.parse_args()
    
    # APIキーの確認
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not API_KEY:
        print("エラー: Gemini API キーが設定されていません。")
        print("環境変数 GOOGLE_API_KEY を設定してください。")
        return
    
    # Gemini APIの初期化
    if not initialize_genai(API_KEY):
        return
    
    # キーワードリストの取得
    if args.keywords:
        # コマンドラインからキーワードを取得
        keywords = [kw.strip() for kw in args.keywords.split(',')]
    else:
        # Excelファイルからキーワードを抽出
        df = load_excel_data(args.file)
        if df is None:
            return
            
        keywords = extract_keywords_from_excel(df, max_keywords=args.max)
    
    if not keywords:
        print("エラー: 有効なキーワードが見つかりませんでした。")
        return
    
    print(f"処理するキーワード（{len(keywords)}件）: {', '.join(keywords)}")
    
    # 記事の一括生成
    start_time = time.time()
    results = generate_articles_batch(
        keywords, 
        args.file, 
        args.output, 
        args.temp, 
        args.delay
    )
    end_time = time.time()
    
    # 結果の表示
    print("\n" + "="*50)
    print(f"一括記事生成完了！")
    print(f"成功: {results['success']}件")
    print(f"失敗: {results['failed']}件")
    print(f"合計処理時間: {end_time - start_time:.2f}秒")
    print(f"出力ディレクトリ: {os.path.abspath(args.output)}")
    print("="*50)

if __name__ == "__main__":
    main()
