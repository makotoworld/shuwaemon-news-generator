#!/bin/bash

# Pythonパスを設定
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# アプリケーションを起動
uvicorn app:app --host 0.0.0.0 --port $PORT
