import * as fs from "node:fs/promises"

export default async function (props: { id: string }): Promise<string> {
	const id = props.id
	const css = await fs.readFile(`../target/js/islands/${id}.css`, "utf8")
	const js = await fs.readFile(`../target/js/islands/${id}.js`, "utf8")

	return (
		<section>
			<div id={id} />
			<style>{css}</style>
			<script type="module">{js}</script>
		</section>
	)
}
