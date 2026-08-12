# EcoPredict 部署指南

## 架構概覽

```
前端（Cloudflare Pages）  ──fetch──▶  後端（Render）
http://ecopredict.pages.dev            https://ecopredict-api.onrender.com
       frontend/                              api/api_server.py
       index.html                             FastAPI + uvicorn
       config.js   ◀── API URL 設定在這
```

---

## 一、後端部署到 Render

### 1-1. 前置準備

確認 `.gitignore` 已排除：
- `.env`（內含 `CWA_API_TOKEN`）
- `data/raw/*.csv`（每次啟動重新抓）
- `models/best_model.pkl`（每次啟動重新訓練）

> ⚠️ 永遠不要把 API Token 推進 repo。

### 1-2. 推送 GitHub

```bash
cd /Users/zhuangzijin/Desktop/Reverie-Hacks-2026/EcoPredict
git add .
git commit -m "feat: add render.yaml + wrangler.toml + CORS env config"
git push origin main
```

### 1-3. Render Dashboard 操作

1. 前往 [https://dashboard.render.com](https://dashboard.render.com) → **New ＋** → **Web Service**
2. 選 **Connect a GitHub repository** → 選你的 repo
3. Render 會自動讀到 `render.yaml`（Infrastructure as Code 模式）
4. 點 **Apply** 確認後，服務會自動建立並部署

### 1-4. 在 Render 設定環境變數（Secrets）

**Dashboard → ecopredict-api → Environment → Add Environment Variable：**

| Key | Value | 說明 |
|-----|-------|------|
| `CWA_API_TOKEN` | `你的 CWA API 授權碼` | ⚠️ Secret，不公開 |
| `CORS_ALLOW_ORIGINS` | `https://ecopredict.pages.dev` | 部署後填入實際 Pages 網址 |

### 1-5. 確認部署完成

```bash
# 健康檢查
curl https://ecopredict-api.onrender.com/healthz

# 應回傳：
# {"status": "ok", "stations_loaded": 1000, "cors_origins": [...]}
```

> 💡 Render 免費方案閒置 15 分鐘後會 spin down，第一個請求需等 30-60 秒冷啟動。

---

## 二、前端部署到 Cloudflare Pages

### 2-1. 更新 config.js（關鍵步驟）

部署前先把 `PROD_API` 改成實際的 Render URL：

```js
// frontend/config.js 第 7 行
const PROD_API = 'https://ecopredict-api.onrender.com';  // ← 換成你的實際 URL
```

> 取得方式：Render Dashboard → ecopredict-api → Settings → **URL** 欄位

### 2-2. Cloudflare Pages Dashboard 操作

1. 前往 [https://pages.cloudflare.com](https://pages.cloudflare.com) → **Create a project**
2. 選 **Connect to Git** → 選你的 GitHub repo
3. 填入以下設定：

| 設定項目 | 值 |
|---------|-----|
| Project name | `ecopredict` |
| Production branch | `main` |
| **Build command** | （留空，純靜態，無 build step） |
| **Build output directory** | `frontend` |
| Root directory | `/`（預設） |

4. 點 **Save and Deploy**
5. 等待 1-2 分鐘，部署完成後網址為：`https://ecopredict.pages.dev`

### 2-3. 驗證

1. 開啟 `https://ecopredict.pages.dev`
2. 開啟瀏覽器 DevTools → Network → 確認 `/api/risk-map` 回傳 200
3. 手機瀏覽器打開網址，確認 responsive 版面正常（右下角 🚨 按鈕）

---

## 三、本地開發（兩個 terminal）

```bash
# Terminal 1：後端（Port 8000）
cd /Users/zhuangzijin/Desktop/Reverie-Hacks-2026/EcoPredict
.venv/bin/uvicorn api.api_server:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2：前端（Port 8080）
cd /Users/zhuangzijin/Desktop/Reverie-Hacks-2026/EcoPredict/frontend
python3 -m http.server 8080
```

然後開啟 `http://127.0.0.1:8080`

> `config.js` 的 `isLocal` 判斷為 `true` → 自動使用 `http://127.0.0.1:8000`

---

## 四、CORS 設定說明

```
環境變數 CORS_ALLOW_ORIGINS 的值範例：

開發階段（允許所有）：
  *

正式部署（只允許自己的 Pages）：
  https://ecopredict.pages.dev

多網域（例如有 preview 環境）：
  https://ecopredict.pages.dev,https://ecopredict-preview.pages.dev
```

---

## 五、部署後更新流程

```bash
# 1. 本地修改程式碼
# 2. 推送到 GitHub
git add . && git commit -m "fix: xxx" && git push origin main

# Render 會自動重新部署後端
# Cloudflare Pages 會自動重新部署前端（通常 < 60 秒）
```
