# しゅわえもんニュース生成システム

YouTubeチャンネル「しゅわえもん」のコンテンツを分析し、Gemini AIを使って記事を自動生成するシステムです。

## 機能

- YouTubeチャンネルからのデータ抽出・解析
- Gemini APIを使用した記事の自動生成
- Web UIとCLIの両方のインターフェース
- 生成コストとトークン使用量の追跡

## 必要条件

- Python 3.10以上
- Google API Key (Gemini APIとYouTube Data API)

## インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/shuwaemon-news-generator.git
cd shuwaemon-news-generator
```

### 2. 仮想環境のセットアップ

```bash
python -m venv venv
source venv/bin/activate  # Linuxの場合
venv\Scripts\activate     # Windowsの場合
```

### 3. 依存パッケージのインストール

```bash
pip install -e .
# または
pip install -r requirements.txt
```

### 4. 環境設定

`.env.example`ファイルを`.env`にコピーし、必要な設定を行います：

```bash
cp .env.example .env
```

`.env`ファイルを編集して、Google API Keyなどを設定します。

## 使用方法

### 1. YouTubeデータの抽出

```bash
python youtube_data_extractor.py
```

これにより`data/shuwaemon_news_data.xlsx`が作成されます。

### 2. 記事生成（CLI）

```bash
python gemini_article_generator.py "キーワード" --temp 0.7
```

オプション：
- `--file` : データファイルのパス指定（デフォルト: data/shuwaemon_news_data.xlsx）
- `--temp` : 生成の多様性（0.0-1.0の間、デフォルト: 0.7）

### 3. WebアプリケーションでのUI起動

```bash
uvicorn app:app --reload
```

ブラウザで http://localhost:8000 にアクセスします。

## ディレクトリ構造

```
shuwaemon-news-generator/
├── app.py                      # FastAPI Webアプリケーション
├── gemini_article_generator.py # CLI版記事生成ツール
├── youtube_data_extractor.py   # YouTubeデータ抽出ツール
├── setup.py                    # パッケージ設定
├── requirements.txt            # 依存パッケージリスト
├── lib/                        # 共通ライブラリ
│   ├── __init__.py
│   ├── gemini_client.py        # Gemini API関連コード
│   └── data_utils.py           # データ処理関連コード
├── tests/                      # テストコード
├── static/                     # 静的ファイル
├── templates/                  # HTMLテンプレート
├── data/                       # データファイル保存ディレクトリ
└── generated_articles/         # 生成された記事の保存先
```

## 環境設定（.env）

```
# Google API Key (Gemini API と YouTube API の両方に使用)
GOOGLE_API_KEY=your_api_key_here

# 認証設定 (オプション - デフォルトは admin/password)
AUTH_USERNAME=admin
AUTH_PASSWORD=secure_password
```

## API使用量とコスト

このシステムはGemini 2.0 Flash Lite APIを使用しています。料金は以下の通りです：

- 入力トークン: $0.0003 / 1000トークン
- 出力トークン: $0.0006 / 1000トークン

API使用履歴は`api_usage_history.json`に保存され、WebUIでも確認できます。

## ライセンス

MIT

## 謝辞

- [Google Gemini API](https://ai.google.dev/)
- [YouTube Data API](https://developers.google.com/youtube/v3)
- [FastAPI](https://fastapi.tiangolo.com/)
