import type { PropsWithChildren } from "@kitajs/html"
import { generateCss } from "../utils/css.ts"

export function Headers(
	props: PropsWithChildren<{
		body: string
		css?: string[]
		title?: string
	}>,
) {
	const css = props.css ?? []
	css.push("src/style.css")
	return (
		<>
			<meta charset="UTF-8" />
			<meta name="viewport" content="width=device-width" />
			<style>{generateCss(props.body, css)}</style>
			<title>{props.title}</title>
		</>
	)
}
