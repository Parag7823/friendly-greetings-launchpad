// Environment-based configuration for API endpoints
export const config = {
  // API base URL - defaults to production, can be overridden with VITE_API_URL
  apiUrl: import.meta.env.VITE_API_URL || 'https://friendly-greetings-launchpad-amey.onrender.com',
  
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
