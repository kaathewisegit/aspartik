import "./pdoc.css"
import type {
	ClassType,
	FunctionType,
	ModuleType,
	VariableType,
} from "../schema"
import highlight from "../utils/highlight"

export function Module(props: ModuleType): JSX.Element {
	return (
		<section class="mx-auto max-w-200" id={props.name}>
			<h1>{props.fullname}</h1>
			<Docstring docstring={props.docstring} />

			{props.classes.map((cls) => (
				<Class {...cls} />
			))}
		</section>
	)
}

function Class(props: ClassType): JSX.Element {
	const title = (
		<span class="title font-mono">
			<span class="italic">class</span>{" "}
			<span>{props.name}</span>:
		</span>
	)

	return (
		<section
			class="relative mx-auto mb-6 max-w-200"
			id={props.name}
		>
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

function Variable(props: VariableType): JSX.Element {
	let annotation = null
	if (props.annotation) {
		annotation = (
			<span class="text-gray-600">: {props.annotation}</span>
		)
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
	const vars = props.list.filter(
		(variable) => !variable.name.startsWith("_"),
	)

	return (
		<>
			{vars.map((variable) => (
				<Variable {...variable} />
			))}
		</>
	)
}

function Func(props: FunctionType): JSX.Element {
	const title = (
		<pre class="overflow-x-scroll font-mono">
			<span class="italic">{props.def}</span> {props.name}
			{props.signature}
		</pre>
	)

	return (
		<section id={props.qualname} class="relative my-2">
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
			class="absolute top-0 -left-8 h-8 w-8 text-center text-2xl opacity-0 transition duration-200 hover:opacity-100"
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
	return (
		<h2 class="flex flex-row bg-gray-200 px-4 py-2">
			{props.title}
		</h2>
	)
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
			<input
				class="peer hidden"
				id={inputId}
				type="checkbox"
			/>
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
			<Source
				source={props.source}
				source_lines={props.source_lines}
			/>
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

function Docstring(props: { docstring: string | null }): JSX.Element | null {
	if (props.docstring) {
		return (
			<div class="my-2 pl-2 text-base/7">
				{props.docstring}
			</div>
		)
	} else {
		return null
	}
}
