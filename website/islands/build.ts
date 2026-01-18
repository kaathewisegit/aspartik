import tailwindcss from "@tailwindcss/vite"
import { build, defineConfig } from "vite"
import solid from "vite-plugin-solid"

const makeConfig = (name: string, entry: string) =>
	defineConfig({
		plugins: [solid(), tailwindcss()],
		build: {
			rollupOptions: {
				input: {
					[name]: entry,
				},
				output: {
					manualChunks: undefined,
					inlineDynamicImports: true,
					entryFileNames: "[name].js",
					assetFileNames: "[name].[ext]",
				},
			},

			emptyOutDir: false,
			outDir: "../../target/js/islands",
		},
	})

const ENTRIES = {
	kernelsBeast: "src/kernelsBeast.tsx",
	kernelsB3: "src/kernelsB3.tsx",
}

for (const [name, entry] of Object.entries(ENTRIES)) {
	await build(makeConfig(name, entry))
}
