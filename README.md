# しゅわえもんニュース生成システム

[![CI/CD Pipeline](https://github.com/yourusername/shuwaemon-news-generator/actions/workflows/cicd.yml/badge.svg)](https://github.com/yourusername/shuwaemon-news-generator/actions/workflows/cicd.yml)
[![codecov](https://codecov.io/gh/yourusername/shuwaemon-news-generator/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/shuwaemon-news-generator)

YouTubeチャンネル「しゅわえもん」のニュース記事をGoogle Gemini APIを使って自動生成するウェブアプリケーション。

## 機能

- キーワードベースでのニュース記事生成
- Excelデータからの関連コンテンツ抽出
- API使用量と料金の追跡
- しゅわえもんスタイルの記事フォーマット

## プロジェクト構造

```
shuwaemon-news-generator/
├── app.py                      # メインアプリケーション
├── setup.py                    # パッケージ設定
├── requirements.txt            # 依存関係
├── .github/
│   └── workflows/
│       └── cicd.yml            # GitHub Actions設定
├── tests/
│   ├── __init__.py
│   ├── test_app.py             # アプリケーションテスト
│   └── conftest.py             # pytestの設定
├── static/
│   └── css/
│       └── styles.css          # スタイルシート
├── templates/                  # Jinja2テンプレート
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   ├── history.html
│   └── error.html
├── generated_articles/         # 生成された記事
└── data/
    └── shuwaimon_news_data.xlsx # ソースデータ
```

## 開発環境のセットアップ

### 前提条件

- Python 3.8以上
- pipまたはpipenv

### インストール手順

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/shuwaemon-news-generator.git
cd shuwaemon-news-generator

# 依存関係のインストール
pip install -e ".[dev]"
# または: 
# pip install -r requirements.txt
# pip install pytest pytest-cov flake8

# 環境変数の設定
export GOOGLE_API_KEY="your-api-key"  # LinuxまたはmacOS
# または
# set GOOGLE_API_KEY=your-api-key  # Windows
```

### 開発サーバーの起動

```bash
uvicorn app:app --reload
```

アプリケーションは `http://localhost:8000` で実行されます。

## テスト

```bash
# 単体テストの実行
pytest

# カバレッジレポート付きテスト
pytest --cov=. --cov-report=term
```

## デプロイ

このアプリケーションはGitHub Actionsを使用して[Render.com](https://render.com)に自動デプロイされます。

1. Renderでウェブサービスを作成
2. 環境変数`GOOGLE_API_KEY`を設定
3. GitHubリポジトリの`Secrets`に以下を設定:
   - `RENDER_API_KEY`: RenderのAPIキー
   - `RENDER_SERVICE_ID`: デプロイするサービスのID

## 使用技術

- FastAPI - 高速なウェブフレームワーク
- Jinja2 - テンプレートエンジン
- Google Gemini API - AIテキスト生成
- Pandas - データ処理
- pytest - テスト自動化
- GitHub Actions - CI/CD

## ライセンス

MIT

## 作者

Makoto Nozaki
