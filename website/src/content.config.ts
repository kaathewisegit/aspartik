import { defineCollection } from "astro:content"
import * as path from "node:path"
import type { LoaderContext } from "astro/loaders"
import { $, Glob } from "bun"

const htmlLoader = {
	name: "pdoc-loader",
	load: async (context: LoaderContext): Promise<void> => {
		const store = context.store

		$`uv run -m python.toolkit pdoc`.cwd("..").quiet()

		const base = "../target/pdoc/"
		const files = new Glob("**/*.html").scan(base)
		for await (const filePath of files) {
			if (filePath === "index.html") {
				continue
			}

			const html = await Bun.file(
				path.join(base, filePath),
			).text()
			store.set({ id: filePath, data: { html } })
		}
	},
}

const reference = defineCollection({
	loader: htmlLoader,
})

export const collections = { reference }
