import Html from "../../../components/html"
import { Module } from "../../../components/pdoc"
import css from "../../../components/pdoc.css"
import modules from "../../../reference"

export async function getStaticParams() {
	const out = []
	for (const module of modules) {
		const modulePath = module.id.replaceAll(".", "/")
		out.push({
			slug: `${modulePath}/index`,
		})
	}
	return out
}

export default function (props: any) {
	const moduleName = props.slug.replace(/\/index$/, "").replaceAll("/", ".")
	const module = modules.find((mod) => mod.fullname === moduleName)
	if (!module) {
		throw new Error("No such module")
	}

	return (
		<Html>
			<style>{css}</style>
			<Module {...module} />
		</Html>
	)
}
