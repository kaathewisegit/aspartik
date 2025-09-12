import { defineConfig } from "@kaathewise/ssg"

export default defineConfig({
	pagesDir: "src/pages/",
	sourceDir: "src/",
	assetDir: "public/",
	outputDir: "../target/website/",
})
