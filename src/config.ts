// Environment-based configuration for API endpoints
const resolveDefaultApiUrl = () => {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }

  if (import.meta.env.DEV) {
    // Local development fallback when running Vite dev server
    return 'http://localhost:8000';
  }

  // When deployed (e.g. on Railway) the frontend and backend share the same origin.
  if (typeof window !== 'undefined' && window.location?.origin) {
    return window.location.origin;
  }

  // Final fallback for non-browser contexts
  return 'http://localhost:8000';
};

export const config = {
  supabase: {
    url: import.meta.env.VITE_SUPABASE_URL || '',
    anonKey: import.meta.env.VITE_SUPABASE_ANON_KEY || '',
  },
  api: {
    baseUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  },
  websocket: {
    // CRITICAL FIX: Configurable polling interval for different environments
    pollingInterval: parseInt(import.meta.env.VITE_POLLING_INTERVAL || '1500', 10), // milliseconds
    reconnectAttempts: parseInt(import.meta.env.VITE_WS_RECONNECT_ATTEMPTS || '5', 10),
    reconnectBaseDelay: parseInt(import.meta.env.VITE_WS_RECONNECT_BASE_DELAY || '1000', 10), // milliseconds
  },
  // API base URL - defaults to same-origin in production, can be overridden with VITE_API_URL
  apiUrl: resolveDefaultApiUrl(),
  
  // WebSocket URL derived from API URL
  get wsUrl() {
    const apiUrl = new URL(this.apiUrl);
    const protocol = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${apiUrl.host}`;
  },
  
  // Environment detection
  isDevelopment: import.meta.env.DEV,
  isProduction: import.meta.env.PROD,
  
  // Other config
  enableAnalytics: import.meta.env.VITE_ENABLE_ANALYTICS === 'true',
  enableDebugLogs: import.meta.env.VITE_DEBUG_LOGS === 'true' || import.meta.env.DEV,
};

// Development overrides
if (config.isDevelopment) {
  console.log('🔧 Development mode detected');
  console.log('📡 API URL:', config.apiUrl);
  console.log('🔌 WS URL:', config.wsUrl);
}
