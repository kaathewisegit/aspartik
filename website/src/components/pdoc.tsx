import "./pdoc.css"
import { For, type JSXElement } from "solid-js"
import type {
	ClassType,
	FunctionType,
	ModuleType,
	VariableType,
} from "../schema"
import highlight from "../utils/highlight"

export function Module(props: ModuleType): JSXElement {
	return (
		<section class="mx-auto max-w-200" id={props.name}>
			<h1>{props.fullname}</h1>
			<Docstring docstring={props.docstring} />

			<For each={props.classes}>
				{(cls, _) => <Class {...cls} />}
			</For>
		</section>
	)
}

function Class(props: ClassType): JSXElement {
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

				<Variables {...props.class_variables} />
				<Variables {...props.instance_variables} />

				<Funcs {...props.staticmethods} />
				<Funcs {...props.classmethods} />
				<Funcs {...props.methods} />
			</div>
		</section>
	)
}

function Variable(props: VariableType): JSXElement {
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

function Variables(props: VariableType[]): JSXElement {
	const varsArr = Array.from(props)
	const vars = varsArr.filter(
		(variable) => !variable.name.startsWith("_"),
	)

	return (
		<For each={vars}>
			{(variable, _) => <Variable {...variable} />}
		</For>
	)
}

function Func(props: FunctionType): JSXElement {
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

function Funcs(props: FunctionType[]): JSXElement {
	const varsArr = Array.from(props)
	const vars = varsArr.filter(
		(variable) => !variable.name.startsWith("_"),
	)

	return <For each={vars}>{(func, _) => <Func {...func} />}</For>
}

function Ref(props: { qualname: string }): JSXElement {
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
	title: JSXElement
	object: VariableType | FunctionType | ClassType
}): JSXElement {
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

function HeaderBare(props: { title: JSXElement }): JSXElement {
	return (
		<h2 class="flex flex-row bg-gray-200 px-4 py-2">
			{props.title}
		</h2>
	)
}

function HeaderCode(props: {
	title: JSXElement
	qualname: string
	source: string
	source_lines: [number, number]
}): JSXElement {
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
}): JSXElement {
	const code = highlight(props.source, "python")

	return (
		<pre
			innerHTML={code}
			class="overflow-x-scroll bg-gray-100 pl-2 font-mono peer-not-checked:hidden"
		/>
	)
}

function Docstring(props: { docstring: string | null }): JSXElement | null {
	if (props.docstring) {
		return (
			<div
				class="my-2 pl-2 text-base/7"
				innerHTML={props.docstring}
			></div>
		)
	} else {
		return null
	}
}
