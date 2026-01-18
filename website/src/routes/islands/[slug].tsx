import * as fs from "node:fs/promises"

export function getStaticParams(): { slug: string }[] {
	const NAMES = ["kernelsBeast", "kernelsB3"]
	const out = []

	for (const name of NAMES) {
		out.push(`${name}.js`)
		out.push(`${name}.css`)
	}

	return out.map((name) => {
		return {
			slug: name,
		}
	})
}

export function getContentType(props: { slug: string }): string {
	if (props.slug.endsWith(".js")) {
		return "application/javascript"
	} else if (props.slug.endsWith(".css")) {
		return "text/css"
	} else {
		return "text/plain"
	}
}

export default async function (props: { slug: string }): Promise<string> {
	const content = await fs.readFile(
		`../target/js/islands/${props.slug}`,
		"utf8",
	)

	return content
}
