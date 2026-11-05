import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/data': 'http://127.0.0.1:8000',
      '/overview': 'http://127.0.0.1:8000',
      '/eda': 'http://127.0.0.1:8000',
      '/regression': 'http://127.0.0.1:8000',
      '/reliability': 'http://127.0.0.1:8000',
      '/forecast': 'http://127.0.0.1:8000',
      '/alerts': 'http://127.0.0.1:8000',
      '/insights': 'http://127.0.0.1:8000',
      '/budget': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/task': 'http://127.0.0.1:8000',
    },
  },
});
