import md2html from "../utils/md2html"

export default async function (props: {
	raw: string
	class?: string
}): Promise<string> {
	const rendered = await md2html(props.raw)

	return <section class={props.class}>{rendered}</section>
}
