import type { JSXElement } from "solid-js"

export function Container(props: { title: string; href: string }): JSXElement {
	return (
		<div class="flex h-48 items-center justify-center border lg:w-128">
			<h2 class="text-xl lg:text-3xl">
				<a href={props.href}>{props.title}</a>
			</h2>
		</div>
	)
}
