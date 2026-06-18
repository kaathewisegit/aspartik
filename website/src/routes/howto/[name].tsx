import * as fs from "node:fs/promises"

import { Headers } from "../../components/html.tsx"
import Markdown from "../../components/markdown.tsx"

export async function getStaticParams() {
	const out = []
	for (const entry of await fs.readdir("pages/howto/")) {
		const name = entry.slice(0, -3)
		if (name === "index") {
			continue
		}
		out.push({ name })
	}
	return out
}

export function Head(_: undefined, body: string) {
	return <Headers body={body} title="How-to guides" />
}

export async function Body(props: { name: string }) {
	const source = await fs.readFile(`pages/howto/${props.name}.dj`, "utf-8")

	return <Markdown raw={source} class="prose mx-auto max-w-200" />
}
