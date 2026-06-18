import { readFile } from "node:fs/promises"
import { Generator } from "@kaathewise/nuclearcss"
import WIND4 from "@kaathewise/nuclearcss/wind4"

export async function generateCss(
	body: string,
	cssPaths: string[],
): Promise<string> {
	const generator = Generator.from_options({ presets: [WIND4] })
	generator.addContent(body)
	for (const cssPath of cssPaths) {
		const css = await readFile(cssPath, "utf8")
		generator.addCSS(css)
	}

	return generator.generate()
}
