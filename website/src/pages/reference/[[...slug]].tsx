import Html from "../../components/html"
import Markdown from "../../components/markdown"
import type {
	ClassType,
	CommonType,
	FunctionType,
	ModuleType,
	VariableType,
} from "../../content/reference"
import modules from "../../content/reference"
import { convertCSS } from "../../utils/css"
import highlight from "../../utils/highlight"

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

export default function (props: { slug: string }) {
	const moduleName = props.slug.replace(/\/index$/, "").replaceAll("/", ".")
	const module = modules.find((mod) => mod.fullname === moduleName)
	if (!module) {
		throw new Error("No such module")
	}

	const css = convertCSS("src/pages/reference/pdoc.css")

	return (
		<Html title={`${module.fullname} reference`}>
			<style>{css}</style>
			<Topbar />
			<Sidebar {...module} />
			<Body {...module} />

			<script type="module">{CHECKBOX_SCRIPT}</script>
		</Html>
	)
}

const CHECKBOX_SCRIPT = `
document.addEventListener("DOMContentLoaded", function () {
	const toggleCheckbox = document.getElementById("sidebar-toggle")

	const navLinks = document.querySelectorAll("nav a")

	navLinks.forEach((link) => {
		link.addEventListener("click", function () {
			if (toggleCheckbox) {
				toggleCheckbox.checked = false
			}
		})
	})
})
`

function moduleUrl(module: { fullname: string }): string {
	return `/reference/${module.fullname.replaceAll(".", "/")}`
}

function Topbar(): JSX.Element {
	return (
		<header class="peer sticky top-0 z-10 flex h-8 flex-row items-center bg-gray-300 pl-2 lg:hidden">
			<input type="checkbox" id="sidebar-toggle" hidden />
			<label for="sidebar-toggle">
				<BurgerIcon />
			</label>
		</header>
	)
}

// TODO: a proper icon library (or a helper module)
function BurgerIcon(): JSX.Element {
	return (
		<svg
			xmlns="http://www.w3.org/2000/svg"
			width="24"
			height="24"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			stroke-width="1.5"
			stroke-linecap="square"
			aria-hidden="true"
		>
			<path d="M4 6l16 0" />
			<path d="M4 12l16 0" />
			<path d="M4 18l16 0" />
		</svg>
	)
}

function Sidebar(props: ModuleType): JSX.Element {
	const LocalRef = (local: CommonType) => `#${local.name}`

	function Refs(): JSX.Element {
		return (
			<div>
				<RefList name="Classes" values={props.classes} href={LocalRef} />
				<RefList name="Functions" values={props.functions} href={LocalRef} />
				<RefList name="Variables" values={props.variables} href={LocalRef} />

				<RefList name="Submodules" values={props.submodules} href={moduleUrl} />
			</div>
		)
	}

	return (
		<nav class="fixed z-10 hidden h-screen w-screen w-screen overflow-y-auto bg-white pl-4 peer-has-[input:checked]:block lg:mt-6 lg:block lg:w-64">
			<Refs />
		</nav>
	)
}

export function RefList<T extends { name: string }>(props: {
	name: string
	values: T[]
	href: (obj: T) => string
}): JSX.Element {
	if (props.values.length === 0) {
		return ""
	}

	function Ref(obj: T): JSX.Element {
		return (
			<li>
				<a href={props.href(obj)}>{obj.name}</a>
			</li>
		)
	}

	return (
		<>
			<h3 class="mb-1 font-bold text-xl">{props.name}</h3>
			<ul class="mb-6 space-y-2">{props.values.map((obj) => Ref(obj))}</ul>
		</>
	)
}

export async function Body(props: ModuleType): Promise<string> {
	return (
		<article id={props.name} class="mx-auto max-w-200 p-4">
			<ModuleHeading {...props} />
			<Docstring docstring={props.docstring} />

			{props.classes.map((cls) => (
				<Class {...cls} />
			))}
			<Variables list={props.variables} />
			<Funcs list={props.functions} />
		</article>
	)
}

function ModuleHeading(props: ModuleType): JSX.Element {
	const fullname = props.fullname
	const modules = []
	const moduleNames = fullname.split(".")
	for (let i = 0; i < moduleNames.length - 1; i += 1) {
		const fullname = moduleNames.slice(0, i + 1).join(".")
		const name = moduleNames[i]
		modules.push({ fullname, name })
	}

	return (
		<h1 class="mb-4 font-bold font-mono text-lg lg:text-4xl">
			{modules.map((module) => (
				<>
					<a href={moduleUrl(module)}>{module.name}</a>.
				</>
			))}

			{props.name}
		</h1>
	)
}

async function Class(props: ClassType): Promise<string> {
	const title = (
		<span class="title font-mono">
			<span class="italic">class</span> <span>{props.name}</span>:
		</span>
	)

	return (
		<section class="relative mx-auto mb-12 max-w-200" id={props.name}>
			<Header object={props} title={title} />

			<Ref qualname={props.qualname} />

			<div class="ml-4">
				<Docstring docstring={props.docstring} />

				<Variables list={props.class_variables} />
				<Variables list={props.instance_variables} />

				<Funcs list={props.staticmethods} />
				<Funcs list={props.classmethods} />
				<Funcs list={props.methods} />
			</div>
		</section>
	)
}

async function Variable(props: VariableType): Promise<string> {
	let annotation = null
	if (props.annotation) {
		annotation = <span class="text-gray-600">: {props.annotation}</span>
	}

	const title = (
		<span class="font-mono">
			{props.name}
			{annotation}
		</span>
	)

	return (
		<section id={props.qualname} class="relative my-2">
			<Header object={props} title={title} />
			<Ref qualname={props.qualname} />
			<Docstring docstring={props.docstring} />
		</section>
	)
}

function Variables(props: { list: VariableType[] }): JSX.Element {
	const vars = props.list.filter((variable) => !variable.name.startsWith("_"))

	return (
		<>
			{vars.map((variable) => (
				<Variable {...variable} />
			))}
		</>
	)
}

async function Func(props: FunctionType): Promise<string> {
	const title = (
		<pre class="overflow-x-scroll font-mono">
			<span class="italic">{props.def}</span> {props.name}
			{props.signature}
		</pre>
	)

	return (
		<section id={props.qualname} class="relative">
			<Header object={props} title={title} />
			<Ref qualname={props.qualname} />
			<Docstring docstring={props.docstring} />
		</section>
	)
}

function Funcs(props: { list: FunctionType[] }): JSX.Element {
	const funcs = props.list.filter((func) => !func.name.startsWith("_"))

	return (
		<>
			{funcs.map((func) => (
				<Func {...func} />
			))}
		</>
	)
}

function Ref(props: { qualname: string }): JSX.Element {
	return (
		<a
			class="-left-8 absolute top-0 h-8 w-8 text-center text-2xl opacity-0 transition duration-200 hover:opacity-100"
			href={`#${props.qualname}`}
		>
			#
		</a>
	)
}

function Header(props: {
	title: JSX.Element
	object: VariableType | FunctionType | ClassType
}): JSX.Element {
	const obj = props.object
	if (obj.source && obj.source_lines) {
		return (
			<HeaderCode
				title={props.title}
				qualname={obj.qualname}
				source={obj.source}
				source_lines={obj.source_lines}
			/>
		)
	} else {
		return <HeaderBare title={props.title} />
	}
}

function HeaderBare(props: { title: JSX.Element }): JSX.Element {
	return <h2 class="flex flex-row bg-gray-200 px-4 py-2">{props.title}</h2>
}

function HeaderCode(props: {
	title: JSX.Element
	qualname: string
	source: string
	source_lines: [number, number]
}): JSX.Element {
	const inputId = `${props.qualname}-view-source`

	return (
		<>
			<input class="peer hidden" id={inputId} type="checkbox" />
			<h2 class="flex flex-row bg-gray-200 px-4 py-2">
				{props.title}

				<label
					class="float-right ml-auto cursor-pointer select-none"
					for={inputId}
				>
					<span class="triangle mr-1">▶</span>
					View source
				</label>
			</h2>
			<Source source={props.source} source_lines={props.source_lines} />
		</>
	)
}

function Source(props: {
	source: string
	source_lines: [number, number]
}): JSX.Element {
	const code = highlight(props.source, "python")

	return (
		<pre class="overflow-x-scroll bg-gray-100 pl-2 font-mono peer-not-checked:hidden">
			{code}
		</pre>
	)
}

async function Docstring(props: { docstring: string | null }): Promise<string> {
	if (props.docstring) {
		return (
			<Markdown
				raw={props.docstring}
				class="prose mt-2 mb-6 pl-2 text-base/7"
			/>
		)
	} else {
		return ""
	}
}
