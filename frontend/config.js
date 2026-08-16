// EcoPredict - API Endpoint Configuration
const IS_LOCAL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const LOCAL_API = 'http://127.0.0.1:8000';
const PROD_API = 'https://ecopredict-backend.onrender.com';

window.ECOPREDICT_API = IS_LOCAL ? LOCAL_API : PROD_API;
