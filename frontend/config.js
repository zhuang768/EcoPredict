/**
 * EcoPredict — API 端點設定
 *
 * 本地開發（localhost / 127.0.0.1）→ 自動指向 http://127.0.0.1:8000
 * Cloudflare Pages 或其他域名     → 指向 Render 生產 API
 *
 * 部署到 Render 後，把 PROD_API 改成你的實際 Render URL，例如：
 *   https://ecopredict-api.onrender.com
 */
(function () {
  const PROD_API = 'https://ecopredict-kpvw.onrender.com'; // ← 部署後換成實際 URL
  const DEV_API  = 'http://127.0.0.1:8000';

  const isLocal = (
    window.location.hostname === 'localhost' ||
    window.location.hostname === '127.0.0.1' ||
    window.location.hostname === ''
  );

  window.ECOPREDICT_API = isLocal ? DEV_API : PROD_API;
})();
