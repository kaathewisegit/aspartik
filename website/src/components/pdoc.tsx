import "../styles/pdoc.css"
import { For, type JSXElement } from "solid-js"
import type { ClassType, ModuleType } from "../schema"

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

	let header: JSXElement
	if (props.source && props.source_lines) {
		header = (
			<HeaderCode
				title={title}
				name={props.name}
				source={props.source}
				source_lines={props.source_lines}
			/>
		)
	} else {
		header = <HeaderBare title={title} />
	}

	return (
		<section class="mx-auto max-w-200" id={props.name}>
			{header}

			<Docstring docstring={props.docstring} />
		</section>
	)
}

function HeaderBare(props: { title: JSXElement }): JSXElement {
	return <h2 class="bg-gray-200 px-4 py-2">{props.title}</h2>
}

function HeaderCode(props: {
	title: JSXElement
	name: string
	source: string
	source_lines: [number, number]
}): JSXElement {
	const inputId = `${props.name}-view-source`

	return (
		<>
			<input
				class="peer hidden"
				id={inputId}
				type="checkbox"
			/>
			<h2 class="bg-gray-200 px-4 py-2">
				{props.title}

				<label
					class="float-right cursor-pointer select-none"
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
	return (
		<pre class="overflow-x-scroll bg-gray-100 pl-2 font-mono peer-not-checked:hidden">
			{props.source}
		</pre>
	)
}

function Docstring(props: { docstring: string | null }): JSXElement | null {
	if (props.docstring) {
		return (
			<section
				class="m-2 pl-2 text-base/7"
				innerHTML={props.docstring}
			></section>
		)
	} else {
		return null
	}
}
