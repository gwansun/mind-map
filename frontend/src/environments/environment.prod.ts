export const environment = {
  production: true,
  apiBaseUrl: '/api',
  graphRefreshInterval: 30000,
  maxVisibleNodes: 500,
  cacheConfig: {
    graphTtl: 30000,
    statsTtl: 60000,
    nodeTtl: 120000,
  },
  debounce: {
    search: 300,
    graphRender: 100,
  },
};
