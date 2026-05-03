import preact from "@preact/preset-vite"
import tailwindcss from "@tailwindcss/vite"
import { build, defineConfig } from "vite"

const makeConfig = (name: string, entry: string) =>
	defineConfig({
		plugins: [preact(), tailwindcss()],
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
