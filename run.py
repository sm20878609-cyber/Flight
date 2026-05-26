#!/usr/bin/env python3
"""
run.py — 一鍵安裝並啟動 AI 智慧航班推薦系統
適用於：電腦上沒有預先安裝 pip 或 streamlit 套件的使用者

使用方式：
    python run.py
"""

import subprocess
import sys
import os

REQUIRED_PACKAGES = [
    "streamlit",
    "pandas",
    "numpy",
    "scikit-learn",
    "plotly",
    "fast-flights",
    "curl_cffi",
    "duckduckgo-search",
]

def ensure_pip():
    """確保 pip 可用"""
    try:
        import pip
        print("✅ pip 已就緒")
    except ImportError:
        print("⚠️  找不到 pip，嘗試自動安裝...")
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ pip 安裝完成")

def install_packages():
    """安裝所有必要套件"""
    print("\n📦 正在檢查並安裝所需套件，請稍候...\n")
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg.replace("-", "_").split("[")[0])
            print(f"  ✅ {pkg} — 已安裝")
        except ImportError:
            print(f"  ⬇️  {pkg} — 安裝中...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                stdout=subprocess.DEVNULL
            )
            print(f"  ✅ {pkg} — 安裝完成")

def launch_app():
    """啟動 Streamlit 應用程式"""
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    if not os.path.exists(app_path):
        print(f"❌ 找不到 app.py，請確認您在正確的資料夾中執行此程式。")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("  🛫 AI 智慧航班機票搜尋與推薦系統 — 啟動中...")
    print("=" * 60)
    print("\n  ✅ 所有套件已就緒，正在啟動網頁介面...")
    print("  🌐 請在瀏覽器開啟：http://localhost:8501")
    print("  🛑 若要停止，請按 Ctrl + C")
    print("=" * 60 + "\n")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ✈️  AI Flight Recommender — 啟動程序")
    print("=" * 60)
    
    ensure_pip()
    install_packages()
    launch_app()
