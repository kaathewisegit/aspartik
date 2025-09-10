import rehypeStringify from "rehype-stringify"
import remarkGfm from "remark-gfm"
import remarkParse from "remark-parse"
import remarkRehype from "remark-rehype"
import { unified } from "unified"

export default async function md2html(md: string) {
	const file = await unified()
		.use(remarkGfm)
		.use(remarkParse)
		.use(remarkRehype)
		.use(rehypeStringify)
		.process(md)

	return String(file)
}
