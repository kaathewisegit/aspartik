import rehypeStringify from "rehype-stringify"
import remarkGfm from "remark-gfm"
import remarkParse from "remark-parse"
import remarkRehype from "remark-rehype"
import { unified } from "unified"

export default async function (props: {
	raw: string
	class?: string
}): Promise<string> {
	const rendered = await md2html(props.raw)

	return <section class={props.class}>{rendered}</section>
}

async function md2html(md: string) {
	const file = await unified()
		.use(remarkGfm)
		.use(remarkParse)
		.use(remarkRehype)
		.use(rehypeStringify)
		.process(md)

	return String(file)
}
