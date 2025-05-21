"""
しゅわえもんニュース生成モデル
実行時エラーを回避するための修正版
"""
import re
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout

# シーケンス長のグローバル設定
SEQ_LENGTH = 5  # 短くして処理しやすくする

def preprocess_text(text):
    """テキストの前処理"""
    # 特殊文字の削除と小文字化
    text = re.sub(r'[^\w\s]', '', str(text))
    # 余分な空白の削除
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_sequences(texts, seq_length=SEQ_LENGTH):
    """テキストから入力シーケンスと次の単語のペアを作成"""
    sequences = []
    next_words = []
    
    for text in texts:
        words = text.split()
        # シーケンス長+1よりも長いテキストのみ処理
        if len(words) > seq_length:
            for i in range(len(words) - seq_length):
                sequences.append(' '.join(words[i:i+seq_length]))
                next_words.append(words[i+seq_length])
    
    return sequences, next_words

def build_model(vocab_size, seq_length=SEQ_LENGTH):
    """LSTMモデルを構築"""
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def generate_news(seed_text, model, tokenizer, seq_length=SEQ_LENGTH, num_words=100):
    """与えられたシード文から新しいニュースを生成"""
    generated_text = seed_text
    
    for _ in range(num_words):
        # 入力シーケンスのトークン化
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # シーケンスが短すぎる場合は終了
        if len(token_list) < seq_length:
            break
            
        # シーケンスの長さを調整
        token_list = token_list[-seq_length:]
        
        # シーケンスのパディング
        token_list = pad_sequences([token_list], maxlen=seq_length, padding='pre')
        
        # 次の単語の予測
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probs)
        
        # インデックスから単語へ変換
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        
        # 単語が見つからない場合はスキップ
        if not output_word:
            continue
        
        # 生成テキストに単語を追加
        generated_text += " " + output_word
        
        # 次の予測のためにシード文を更新
        seed_text = ' '.join(seed_text.split()[1:]) + " " + output_word
    
    return generated_text

def generate_structured_news(topic, model, tokenizer, data, seq_length=SEQ_LENGTH):
    """テンプレートを使用して構造化されたニュース記事を生成"""
    try:
        # トピックに関連する既存のニュースを見つける
        vectorizer = TfidfVectorizer()
        
        # コンバインドテキストがない場合は作成
        if 'combined_text' not in data.columns:
            data['combined_text'] = data['title'].fillna('') + " " + data['description'].fillna('')
        else:
            # NaN値を空文字列に置換
            data['combined_text'] = data['combined_text'].fillna('')
            
        # データがない場合は空の文字列を返す
        if len(data) == 0:
            return "データがないため記事を生成できません。"
            
        # NaN値を含む行を除外
        data = data.dropna(subset=['combined_text'])
        if len(data) == 0:
            return "有効なテキストデータがないため記事を生成できません。"
            
        tfidf_matrix = vectorizer.fit_transform(data['combined_text'])
        topic_vec = vectorizer.transform([topic])
        
        # コサイン類似度で関連記事を見つける
        cosine_similarities = cosine_similarity(topic_vec, tfidf_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-6:-1]  # 上位5件
        
        # 結果が空の場合
        if len(related_docs_indices) == 0:
            return f"トピック「{topic}」に関連するデータが見つかりませんでした。"
        
        # タイトル生成
        try:
            # 関連記事のタイトルスタイルを分析
            title_patterns = [data.iloc[i]['title'] for i in related_docs_indices]
            
            # ランダムにタイトルパターンを選択
            title_template = random.choice(title_patterns)
            
            # タイトル内の長い単語をトピックに置き換え
            words = title_template.split()
            long_words = [word for word in words if len(word) > 3]
            
            if long_words:
                generated_title = title_template.replace(long_words[0], topic)
            else:
                generated_title = f"{topic}に関する最新ニュース"
        except Exception as e:
            print(f"タイトル生成エラー: {e}")
            generated_title = f"{topic}に関する最新ニュース"
        
        # イントロ部分の生成
        try:
            first_doc_text = data.iloc[related_docs_indices[0]]['combined_text']
            intro_seed = ' '.join(first_doc_text.split()[:seq_length])
            intro_text = generate_news(intro_seed, model, tokenizer, seq_length, 30)
        except Exception as e:
            print(f"イントロ生成エラー: {e}")
            intro_text = f"{topic}について注目されています。"
        
        # 本文の生成
        try:
            body_seed = f"{topic} について"
            if len(body_seed.split()) < seq_length:
                body_seed = f"{topic} について 最近 話題 になっています"
            body_text = generate_news(body_seed, model, tokenizer, seq_length, 100)
        except Exception as e:
            print(f"本文生成エラー: {e}")
            body_text = f"{topic}に関する詳細情報が待たれます。"
        
        # 結論部分の生成
        try:
            conclusion_seed = "まとめると " + " ".join([topic] * seq_length)
            conclusion_text = generate_news(conclusion_seed, model, tokenizer, seq_length, 20)
        except Exception as e:
            print(f"結論生成エラー: {e}")
            conclusion_text = f"今後の{topic}の展開に注目です。"
        
        # 記事の構成
        full_article = f"""
{generated_title}

{intro_text}

{body_text}

{conclusion_text}
        """
        
        return full_article
        
    except Exception as e:
        print(f"記事生成中にエラーが発生しました: {e}")
        return f"トピック「{topic}」の記事生成中にエラーが発生しました。"

# モデル訓練用関数
def train_news_model(data_file, epochs=10, batch_size=32, seq_length=SEQ_LENGTH):
    """
    ニュース生成モデルを訓練する関数
    この関数は独立して呼び出され、実行時エラーを回避します
    """
    print("データの読み込みを開始...")
    try:
        data = pd.read_excel(data_file)
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None, None
    
    print(f"読み込み完了。{len(data)}件のデータが見つかりました。")
    
    # テキストの前処理
    print("テキストの前処理中...")
    data['processed_title'] = data['title'].apply(preprocess_text)
    data['processed_description'] = data['description'].apply(preprocess_text)
    data['combined_text'] = data['processed_title'] + " " + data['processed_description']
    
    # シーケンスの作成
    print("シーケンスデータの作成中...")
    sequences, next_words = create_sequences(data['combined_text'].tolist(), seq_length)
    
    if len(sequences) == 0:
        print("エラー: 有効なシーケンスが見つかりませんでした。")
        print("データ量が少ない、またはテキストが短すぎる可能性があります。")
        return None, None
        
    print(f"作成されたシーケンス数: {len(sequences)}")
    
    # テキストのトークン化
    print("テキストのトークン化中...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    total_words = len(tokenizer.word_index) + 1
    
    print(f"語彙数: {total_words}")
    
    # シーケンスをインデックスに変換
    print("シーケンスの変換とデータの準備中...")
    sequences_indices = tokenizer.texts_to_sequences(sequences)
    X = pad_sequences(sequences_indices, maxlen=seq_length)
    
    # 次の単語をone-hotエンコーディング
    next_words_indices = tokenizer.texts_to_sequences(next_words)
    
    # 無効なシーケンスをフィルタリング
    valid_indices = []
    filtered_X = []
    filtered_next_words_indices = []
    
    for i, word_seq in enumerate(next_words_indices):
        if len(word_seq) > 0:  # 空でないシーケンスのみ保持
            valid_indices.append(i)
            filtered_next_words_indices.append(word_seq)
            filtered_X.append(X[i])
    
    if len(filtered_X) == 0:
        print("エラー: 有効なシーケンスが見つかりませんでした。")
        return None, None
    
    print(f"フィルタリング後の有効なシーケンス数: {len(filtered_X)}")
    
    X = np.array(filtered_X)
    y = np.zeros((len(filtered_next_words_indices), total_words))
    
    # one-hotエンコーディングを手動で実行
    for i, seq in enumerate(filtered_next_words_indices):
        if len(seq) > 0:
            y[i, seq[0]] = 1
    
    # モデルの構築
    print("モデルの構築中...")
    model = build_model(total_words, seq_length)
    
    # モデルのトレーニング
    print(f"モデルのトレーニングを開始... (エポック数: {epochs}, バッチサイズ: {batch_size})")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    
    print("トレーニング完了")
    return model, tokenizer
