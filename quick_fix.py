"""
モデルがない、または不完全な場合でも
テンプレートベースで記事を生成するための応急処置スクリプト
"""
import pandas as pd
import random
import re
import os

def generate_simple_article(topic, data_file='shuwaimon_news_data.xlsx'):
    """
    しゅわえもんニュースのスタイルを模倣した簡易的な記事生成
    モデルを使わず、テンプレートベースのみで実装
    """
    try:
        # データの読み込み
        if not os.path.exists(data_file):
            return f"{topic}に関する記事を生成できませんでした。データファイルがありません。"
            
        data = pd.read_excel(data_file)
        
        # NaN値を空文字列に置換
        data['title'] = data['title'].fillna('')
        data['description'] = data['description'].fillna('')
        
        # データの確認
        if len(data) == 0:
            return f"{topic}に関する記事を生成できませんでした。データがありません。"
        
        # タイトルから選択
        titles = data['title'].tolist()
        titles = [t for t in titles if len(t) > 10]  # 短すぎるタイトルを除外
        
        if not titles:
            generated_title = f"{topic}に関する最新ニュース"
        else:
            base_title = random.choice(titles)
            words = base_title.split()
            if len(words) > 3:
                # 最初のいくつかの単語を保持し、トピックを挿入
                generated_title = ' '.join(words[:2]) + f" {topic} " + ' '.join(words[-2:])
            else:
                generated_title = f"{topic}に関する{words[-1]}"
        
        # 概要文から選択
        descriptions = data['description'].tolist()
        descriptions = [d for d in descriptions if isinstance(d, str) and len(d) > 30]  # 有効な説明文のみ
        
        # イントロ部分
        if not descriptions:
            intro = f"{topic}に関する注目すべき最新情報をお届けします。"
        else:
            base_intro = random.choice(descriptions)
            intro_sentences = base_intro.split('。')[:2]
            intro = f"{topic}に関する注目ニュースです。" + '。'.join(intro_sentences) + "。"
        
        # 本文部分
        body_paragraphs = []
        
        # いくつかのランダムな説明文を組み合わせて本文を作成
        if len(descriptions) >= 3:
            selected_desc = random.sample(descriptions, min(3, len(descriptions)))
            for desc in selected_desc:
                # トピックに関連する文章に変換
                new_paragraph = desc.replace('について', f"{topic}について")
                new_paragraph = re.sub(r'([一-龠ぁ-んァ-ン]{2,4})(は|が|を)', f"{topic}\\2", new_paragraph, count=1)
                body_paragraphs.append(new_paragraph)
        else:
            body_paragraphs.append(f"{topic}の詳細については現在調査中です。")
            body_paragraphs.append(f"今後も{topic}に関する最新情報を随時お伝えしていきます。")
        
        # 結論部分
        conclusion = f"今後も{topic}の動向に注目していきましょう。最新情報が入り次第、お伝えします。"
        
        # 記事の組み立て - 修正済み
        body_text = "\n\n".join(body_paragraphs)
        article = f"""
{generated_title}

{intro}

{body_text}

{conclusion}
        """
        
        return article
        
    except Exception as e:
        print(f"記事生成中にエラーが発生しました: {e}")
        return f"{topic}に関する記事を生成できませんでした。エラー: {e}"

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        topic = sys.argv[1]
        print(generate_simple_article(topic))
    else:
        print("使用方法: python quick_fix.py [トピック]")
