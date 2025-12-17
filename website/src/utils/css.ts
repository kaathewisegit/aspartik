import { readFile } from "node:fs/promises"
import tailwindcss from "@tailwindcss/postcss"
import postcss from "postcss"

const processor: postcss.Processor = postcss([tailwindcss()])

export async function convertCSS(path: string): Promise<string> {
	const tailwindSrc = await readFile(path, "utf8")
	const result = await processor.process(tailwindSrc, { from: path })

	return result.css
}
