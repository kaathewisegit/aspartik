import * as fs from "node:fs/promises"

import Html from "../../components/html.tsx"
import Markdown from "../../components/markdown.tsx"

export default async function () {
	const source = await fs.readFile(`pages/howto/index.dj`, "utf-8")

	return (
		<Html title="How-to guides">
			<Markdown raw={source} class="prose mx-auto max-w-200" />
		</Html>
	)
}
