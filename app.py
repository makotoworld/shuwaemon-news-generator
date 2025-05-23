"""
しゅわえもんニュース生成ウェブアプリケーション
FastAPIを使ったGUI実装
"""
import os
from dotenv import load_dotenv
import json
import time
from datetime import datetime
from typing import List, Optional
from pathlib import Path

# 環境変数のロード（開発環境用）
load_dotenv()

from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from contextlib import asynccontextmanager

from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends, HTTPException, status
from secrets import compare_digest

# 共通ライブラリのインポート
from lib.llm_client import get_llm_manager, LLMProvider
from lib.data_utils import load_excel_data, find_relevant_content, create_prompt, get_keyword_suggestions

# アプリケーション設定
APP_TITLE = "しゅわえもんニュース生成システム"
DATA_FILE = "data/shuwaemon_news_data.xlsx"  # Excelデータファイル
HISTORY_FILE = "api_usage_history.json"  # API使用履歴
OUTPUT_DIR = "generated_articles"        # 生成記事の保存ディレクトリ

# Gemini API設定
API_KEY = os.environ.get("GOOGLE_API_KEY")

# 認証機能設定
security = HTTPBasic()

# 環境変数からユーザー名とパスワードを取得
AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "password")

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = compare_digest(credentials.username, AUTH_USERNAME)
    correct_password = compare_digest(credentials.password, AUTH_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="認証に失敗しました",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# ライフスパン定義 - FastAPIアプリの初期化前に配置
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動時の処理
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

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

# LLMマネージャーの初期化
llm_manager = get_llm_manager()

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

def generate_article_with_llm(keyword: str, temperature: float = 0.7, provider: Optional[LLMProvider] = None) -> tuple:
    """
    LLMを使用して記事を生成し、トークン使用量を返す
    """
    try:
        # データフレームの読み込み
        df = load_excel_data(DATA_FILE)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"データファイル '{DATA_FILE}' が空であるか存在しません")

        # 関連コンテンツの検索
        relevant_content = find_relevant_content(df, keyword)

        # プロンプトの作成
        prompt = create_prompt(keyword, relevant_content)

        # 記事生成（LLMマネージャーを使用）
        article, input_tokens, output_tokens, cost, used_provider = llm_manager.generate_article(
            prompt, temperature, provider
        )
        
        return article, input_tokens, output_tokens, cost, used_provider

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
async def index(request: Request, username: str = Depends(get_current_username), history: APIUsageHistory = Depends(get_api_usage_history)):
    # Excel データの読み込み
    df = load_excel_data(DATA_FILE)

    # キーワード候補の取得
    keyword_suggestions = get_keyword_suggestions(df)
    
    # 利用可能なLLMプロバイダーの取得
    available_providers = llm_manager.get_available_providers()
    
    # プロバイダー情報の整理
    provider_info = {
        "gemini": {"name": "Google Gemini", "model": "gemini-2.0-flash-lite", "available": available_providers.get("gemini", False)},
        "openai": {"name": "OpenAI GPT", "model": "gpt-4o-mini", "available": available_providers.get("openai", False)},
        "anthropic": {"name": "Anthropic Claude", "model": "claude-3-haiku", "available": available_providers.get("anthropic", False)}
    }

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "title": APP_TITLE,
            "keyword_suggestions": keyword_suggestions,
            "history": history.history,
            "total_cost": history.total_cost,
            "provider_info": provider_info,
            "available_providers": available_providers
        }
    )

@app.post("/generate")
async def generate_article_endpoint(
    request: Request,
    keyword: str = Form(...),
    temperature: float = Form(0.7),
    provider: Optional[str] = Form(None),
    history: APIUsageHistory = Depends(get_api_usage_history)
):
    try:
        # プロバイダーの型変換
        llm_provider = None
        if provider and provider in ["gemini", "openai", "anthropic"]:
            llm_provider = provider
        
        # 記事の生成
        article, input_tokens, output_tokens, cost, used_provider = generate_article_with_llm(keyword, temperature, llm_provider)

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
                "filename": os.path.basename(filepath),
                "provider": used_provider
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

@app.get("/view-article/{filename}")
async def view_article(request: Request, filename: str):
    """生成された記事の内容を表示"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="記事ファイルが見つかりません")
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            article_content = f.read()
        
        # ファイル名からキーワードと日時を抽出
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('_')
        if len(parts) >= 3:
            keyword = '_'.join(parts[:-2])
            date_part = parts[-2]
            time_part = parts[-1]
            formatted_date = f"{date_part[:4]}/{date_part[4:6]}/{date_part[6:8]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
        else:
            keyword = base_name
            formatted_date = "不明"
        
        return templates.TemplateResponse(
            "article_view.html",
            {
                "request": request,
                "title": APP_TITLE,
                "keyword": keyword,
                "article_content": article_content,
                "filename": filename,
                "formatted_date": formatted_date
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"記事の読み込みエラー: {str(e)}")

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
