export const environment = {
  production: false,
  apiBaseUrl: 'http://localhost:8000',
  graphRefreshInterval: 30000,
  maxVisibleNodes: 500,
  cacheConfig: {
    graphTtl: 30000, // 30 seconds
    statsTtl: 60000, // 60 seconds
    nodeTtl: 120000, // 2 minutes
  },
  debounce: {
    search: 300,
    graphRender: 100,
  },
};
