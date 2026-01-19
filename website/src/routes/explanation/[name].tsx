import * as fs from "node:fs/promises"

import Html from "../../components/html.tsx"
import Markdown from "../../components/markdown.tsx"

export async function getStaticParams() {
	const out = []
	for (const entry of await fs.readdir("pages/explanation/")) {
		const name = entry.slice(0, -3)
		if (name === "index") {
			continue
		}
		out.push({ name })
	}
	return out
}

export default async function (props: { name: string }) {
	const source = await fs.readFile(
		`pages/explanation/${props.name}.dj`,
		"utf-8",
	)

	return (
		<Html title="Explanations">
			<Markdown raw={source} class="prose mx-auto max-w-200" />
		</Html>
	)
}
