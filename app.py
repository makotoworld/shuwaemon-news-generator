"""
しゅわえもんニュース生成ウェブアプリケーション
FastAPIを使ったGUI実装
"""
import os
import json
import time
import pandas as pd
from datetime import datetime
from typing import List, Optional
from pathlib import Path

import google.generativeai as genai
from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from contextlib import asynccontextmanager

# アプリケーション設定
APP_TITLE = "しゅわえもんニュース生成システム"
DATA_FILE = "shuwaimon_news_data.xlsx"  # Excelデータファイル
HISTORY_FILE = "api_usage_history.json"  # API使用履歴
OUTPUT_DIR = "generated_articles"        # 生成記事の保存ディレクトリ

# Gemini API設定
API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = "gemini-1.5-pro"  # または "gemini-1.0-pro"など他のモデル

# ライフスパン定義 - FastAPIアプリの初期化前に配置
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動時の処理
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    if not os.path.exists(DATA_FILE):
        print(f"警告: データファイル '{DATA_FILE}' が見つかりません")
    if not API_KEY:
        print("警告: GOOGLE_API_KEY環境変数が設定されていません")
    
    yield

# FastAPIアプリケーション初期化
app = FastAPI(title=APP_TITLE, lifespan=lifespan)

# 静的ファイルとテンプレートの設定
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ディレクトリの初期化
Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

# Gemini APIの初期化
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("警告: GOOGLE_API_KEY環境変数が設定されていません")

# データモデル
class ArticleRequest(BaseModel):
    keyword: str
    temperature: float = 0.7

class APIUsage(BaseModel):
    timestamp: str
    keyword: str
    tokens_input: int
    tokens_output: int
    cost: float

class APIUsageHistory(BaseModel):
    history: List[APIUsage] = []
    total_cost: float = 0.0

# API料金計算（Gemini 1.5 Proの場合）
def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Gemini APIの使用コストを計算
    注: 料金は変更される可能性があるため公式ドキュメントを参照してください
    """
    # Gemini 1.5 Proの料金（2025年5月現在）
    input_cost_per_1k = 0.0025  # $0.0025 per 1K input tokens
    output_cost_per_1k = 0.0075  # $0.0075 per 1K output tokens
    
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    
    return input_cost + output_cost

# 使用履歴の読み込みと保存
def load_api_usage_history() -> APIUsageHistory:
    """API使用履歴をロード"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
                return APIUsageHistory(**data)
        except Exception as e:
            print(f"履歴ファイルの読み込みエラー: {e}")
    
    return APIUsageHistory()

def save_api_usage(history: APIUsageHistory):
    """API使用履歴を保存"""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history.model_dump(), f, indent=2)

# Excelデータの読み込み
def load_excel_data():
    """データファイルからExcelデータを読み込む"""
    try:
        if os.path.exists(DATA_FILE):
            return pd.read_excel(DATA_FILE)
        else:
            print(f"警告: データファイル '{DATA_FILE}' が見つかりません")
            return pd.DataFrame()
    except Exception as e:
        print(f"Excelファイルの読み込みエラー: {e}")
        return pd.DataFrame()

# キーワード関連の関数
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

def find_relevant_content(df: pd.DataFrame, keyword: str, num_samples: int = 5):
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

# 記事生成関数
def create_prompt(keyword: str, examples_df: pd.DataFrame) -> str:
    """Gemini APIへのプロンプトを作成"""
    # 例を整形
    examples_text = ""
    for i, row in examples_df.iterrows():
        examples_text += f"タイトル: {row['title']}\n"
        examples_text += f"概要: {row['description']}\n\n"
    
    # プロンプトテンプレート
    prompt = f"""
あなたは「しゅわえもんニュース」というYouTubeチャンネルの記事を書くライターです。
以下の特徴を持つニュース記事を日本語で作成してください:

1. キーワード「{keyword}」に関する最新のニュース記事
2. しゅわえもんニュースのスタイルで書かれた
3. 事実ベースで読者が理解しやすい文章
4. 見出し、導入部、本文、結論という構成

以下はしゅわえもんニュースの例です:

{examples_text}

これらの例を参考にして、「{keyword}」についての記事を作成してください。
記事のフォーマットは以下の通りです:

[タイトル]

[導入部 - キーワードについての簡単な紹介と記事の概要]

[本文 - キーワードに関する詳細な内容]

[結論 - 要点のまとめと今後の展望]
"""
    return prompt

async def generate_article_with_gemini(keyword: str, temperature: float = 0.7) -> tuple:
    """
    Gemini APIを使用して記事を生成し、トークン使用量を返す
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Google API Keyが設定されていません")
        
    try:
        # データフレームの読み込み
        df = load_excel_data()
        if df.empty:
            raise HTTPException(status_code=404, detail=f"データファイル '{DATA_FILE}' が空であるか存在しません")
        
        # 関連コンテンツの検索
        relevant_content = find_relevant_content(df, keyword)
        
        # プロンプトの作成
        prompt = create_prompt(keyword, relevant_content)
        
        # モデルの設定
        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
        
        # モデルの作成と記事生成
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt, generation_config=generation_config)
        
        # トークン数の取得（注: 実際のAPIレスポンスによっては異なる可能性があります）
        # ここでは仮の実装として、単語数から概算しています
        input_tokens = len(prompt.split())
        output_tokens = len(response.text.split())
        
        # コスト計算
        cost = calculate_cost(input_tokens, output_tokens)
        
        # 生成された記事とトークン情報を返す
        return response.text, input_tokens, output_tokens, cost
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"記事生成エラー: {str(e)}")

def save_generated_article(keyword: str, article: str) -> str:
    """
    生成された記事をファイルに保存して、ファイルパスを返す
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{keyword.replace(' ', '_')}_{timestamp}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(article)
        return filepath
    except Exception as e:
        print(f"ファイル保存エラー: {e}")
        return ""

# 依存関係
def get_api_usage_history():
    return load_api_usage_history()

# ルート
@app.get("/", response_class=HTMLResponse)
async def index(request: Request, history: APIUsageHistory = Depends(get_api_usage_history)):
    # Excel データの読み込み
    df = load_excel_data()
    
    # キーワード候補の取得
    keyword_suggestions = get_keyword_suggestions(df)
    
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request, 
            "title": APP_TITLE,
            "keyword_suggestions": keyword_suggestions,
            "history": history.history,
            "total_cost": history.total_cost
        }
    )

@app.post("/generate")
async def generate_article(
    request: Request,
    keyword: str = Form(...),
    temperature: float = Form(0.7),
    history: APIUsageHistory = Depends(get_api_usage_history)
):
    try:
        # 記事の生成
        article, input_tokens, output_tokens, cost = await generate_article_with_gemini(keyword, temperature)
        
        # ファイルに保存
        filepath = save_generated_article(keyword, article)
        
        # API使用履歴の更新
        usage = APIUsage(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            keyword=keyword,
            tokens_input=input_tokens,
            tokens_output=output_tokens,
            cost=cost
        )
        
        history.history.append(usage)
        history.total_cost += cost
        save_api_usage(history)
        
        # 出力を整形（HTMLで表示するため）
        formatted_article = article.replace("\n", "<br>")
        
        return templates.TemplateResponse(
            "result.html", 
            {
                "request": request,
                "title": APP_TITLE,
                "keyword": keyword,
                "article": formatted_article,
                "raw_article": article,
                "filepath": filepath,
                "temperature": temperature,
                "tokens_input": input_tokens,
                "tokens_output": output_tokens,
                "cost": cost,
                "total_cost": history.total_cost,
                "filename": os.path.basename(filepath)
            }
        )
    
    except HTTPException as e:
        return templates.TemplateResponse(
            "error.html", 
            {
                "request": request,
                "title": APP_TITLE,
                "status_code": e.status_code,
                "detail": e.detail
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "error.html", 
            {
                "request": request,
                "title": APP_TITLE,
                "status_code": 500,
                "detail": str(e)
            }
        )

@app.get("/download/{filename}")
async def download_article(filename: str):
    """生成された記事をダウンロード"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="ファイルが見つかりません")
    
    return FileResponse(
        filepath, 
        media_type="text/plain", 
        filename=filename
    )

@app.get("/history")
async def view_history(request: Request, history: APIUsageHistory = Depends(get_api_usage_history)):
    """API使用履歴の表示"""
    return templates.TemplateResponse(
        "history.html", 
        {
            "request": request,
            "title": APP_TITLE, 
            "history": history.history,
            "total_cost": history.total_cost
        }
    )

@app.get("/clear-history")
async def clear_history(request: Request):
    """API使用履歴のクリア"""
    # 履歴を新規作成
    new_history = APIUsageHistory()
    save_api_usage(new_history)
    
    return templates.TemplateResponse(
        "history.html", 
        {
            "request": request,
            "title": APP_TITLE, 
            "history": [],
            "total_cost": 0.0,
            "message": "履歴がクリアされました"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # 開発サーバー起動
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
