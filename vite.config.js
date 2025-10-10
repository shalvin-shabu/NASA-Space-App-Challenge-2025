import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    host: true,   // allow LAN
    port: 3000
  }
})