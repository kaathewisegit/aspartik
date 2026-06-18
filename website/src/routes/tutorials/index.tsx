import * as fs from "node:fs/promises"

import { Headers } from "../../components/html.tsx"
import Markdown from "../../components/markdown.tsx"

export function Head(_: undefined, body: string) {
	return <Headers body={body} title="Tutorials" />
}

export async function Body() {
	const source = await fs.readFile(`pages/tutorials/index.dj`, "utf-8")

	return <Markdown raw={source} class="prose mx-auto max-w-200" />
}
