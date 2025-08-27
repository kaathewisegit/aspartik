import solid from "@astrojs/solid-js"
import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from "astro/config"

export default defineConfig({
	vite: {
		plugins: [tailwindcss()],
	},

	integrations: [solid()],

	outDir: "../target/website/",
})
