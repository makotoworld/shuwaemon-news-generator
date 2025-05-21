#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

def extract_data():
    """YouTubeデータ抽出スクリプトを実行"""
    print("YouTubeデータの抽出を開始します...")
    import youtube_data_extractor
    youtube_data_extractor.main()
    
def train_model():
    """ニュース生成モデルの訓練を実行"""
    print("モデルのトレーニングを開始します...")
    
    # データが存在するか確認
    if not os.path.exists('shuwaimon_news_data.xlsx'):
        print("エラー: データファイル(shuwaimon_news_data.xlsx)が見つかりません。")
        print("まず 'python main.py --extract' を実行してデータを抽出してください。")
        return
    
    # news_generator を安全にインポート
    import news_generator
    
    # モデルを訓練
    model, tokenizer = news_generator.train_news_model(
        data_file='shuwaimon_news_data.xlsx',
        epochs=20,
        batch_size=32,
        seq_length=news_generator.SEQ_LENGTH
    )
    
    # モデルが正常に訓練されたかチェック
    if model is None or tokenizer is None:
        print("エラー: モデルの訓練に失敗しました。")
        return
    
    # モデルの保存
    model.save('shuwaimon_news_model.h5')
    
    # トークナイザの保存
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("モデルのトレーニングが完了しました。")
    print("モデルは 'shuwaimon_news_model.h5' として保存されました。")
    print("トークナイザは 'tokenizer.pickle' として保存されました。")

def generate_article(topic):
    """指定されたトピックに関するニュース記事を生成"""
    print(f"トピック「{topic}」に関するニュース記事を生成します...")
    
    # 通常のモデルベース記事生成を試行
    try:
        import news_generator
        
        # データ読み込み
        data = pd.read_excel('shuwaimon_news_data.xlsx')
        
        # モデルとトークナイザが存在するか確認
        if os.path.exists('shuwaimon_news_model.h5') and os.path.exists('tokenizer.pickle'):
            # モデルとトークナイザをロード
            model = load_model('shuwaimon_news_model.h5')
            
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            
            # 記事生成
            article = news_generator.generate_structured_news(
                topic, 
                model, 
                tokenizer, 
                data, 
                news_generator.SEQ_LENGTH
            )
        else:
            # モデルがない場合は簡易生成を使用
            print("モデルが見つからないため、テンプレートベースの簡易記事生成を使用します...")
            from quick_fix import generate_simple_article
            article = generate_simple_article(topic)
        
        # 結果を表示
        print("\n" + "="*50)
        print(article)
        print("="*50)
        
        # ファイルに保存
        output_file = f"generated_article_{topic.replace(' ', '_')}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(article)
        
        print(f"\n記事は '{output_file}' として保存されました。")
        
    except Exception as e:
        print(f"記事生成中にエラーが発生しました: {e}")
        
        # エラーが発生した場合は簡易版を試行
        try:
            print("標準生成に失敗したため、簡易記事生成を試行します...")
            from quick_fix import generate_simple_article
            article = generate_simple_article(topic)
            
            # 結果を表示
            print("\n" + "="*50)
            print(article)
            print("="*50)
            
            # ファイルに保存
            output_file = f"generated_article_{topic.replace(' ', '_')}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(article)
            
            print(f"\n記事は '{output_file}' として保存されました。")
            
        except Exception as e2:
            print(f"簡易記事生成も失敗しました: {e2}")
            print("データまたはモデルに問題がある可能性があります。")

def main():
    """メイン関数: コマンドライン引数の解析と実行"""
    parser = argparse.ArgumentParser(description='しゅわえもんニュース生成システム')
    
    # コマンドライン引数の設定
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--extract', action='store_true', help='YouTubeから動画データを抽出')
    group.add_argument('--train', action='store_true', help='ニュース生成モデルをトレーニング')
    group.add_argument('--generate', type=str, metavar='TOPIC', help='指定されたトピックのニュース記事を生成')
    
    args = parser.parse_args()
    
    # 引数に応じた処理を実行
    if args.extract:
        extract_data()
    elif args.train:
        train_model()
    elif args.generate:
        generate_article(args.generate)

if __name__ == '__main__':
    main()
