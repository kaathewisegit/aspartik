import "../styles/pdoc.css"
import { For, type JSXElement } from "solid-js"
import type { ClassType, ModuleType } from "../schema"

function Docstring(props: { docstring: string | null }): JSXElement | null {
	if (props.docstring) {
		return (
			<section
				class="m-2 pl-2"
				innerHTML={props.docstring}
			></section>
		)
	} else {
		return null
	}
}

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
	const inputId = `${props.name}-view-source`

	return (
		<section class="mx-auto max-w-200" id={props.name}>
			<input
				class="peer hidden"
				id={inputId}
				type="checkbox"
			/>
			<div class="space-x-2 bg-gray-200 px-4 py-2">
				<span>class</span>
				<span>{props.name}</span>
				<label
					class="float-right cursor-pointer"
					for={inputId}
				>
					<span class="triangle">▶</span>
					View source
				</label>
			</div>
			<Code
				class="peer-not-checked:hidden"
				source={props.source}
			/>
			<Docstring docstring={props.docstring} />
		</section>
	)
}

function Code(props: { class?: string; source: string }): JSXElement {
	return <pre class={props.class}>{props.source}</pre>
}
