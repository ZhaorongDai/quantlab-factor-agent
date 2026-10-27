import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The build goes into the Python package, where factor-agent-web serves it; it is committed.
// `npm run dev` serves the page on 5173 and sends /api to a factor-agent-web on its default port.
export default defineConfig({
  plugins: [react()],
  build: { outDir: '../factor_agent/web/static', emptyOutDir: true },
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
})
