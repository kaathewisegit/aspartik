import tailwindcss from "@tailwindcss/postcss"
import postcss from "postcss"

const processor: postcss.Processor = postcss([tailwindcss()])

export async function convertCSS(path: string): Promise<string> {
	const file = Bun.file(path)
	const tailwindSrc = await file.text()
	const result = await processor.process(tailwindSrc, { from: path })

	return result.css
}
