import Html from "../components/html.tsx"

export default function () {
	return (
		<Html
			title="Documentation"
			class="flex min-h-screen items-center justify-center"
		>
			<div class="grid w-full max-w-5xl grid-cols-1 gap-x-8 gap-y-4 md:grid-cols-2">
				<Container
					title="Tutorials"
					href="/tutorials"
					desc="Step by step lessons"
				/>
				<Container
					title="How-to guides"
					href="/howto"
					desc="Solutions to concrete problems"
				/>
				<Container
					title="Explanation"
					href="/explanation"
					desc="Architecture and theory notes"
				/>
				<Container
					title="Reference"
					href="/reference"
					desc="Description of the Python API"
				/>
			</div>
		</Html>
	)
}

export function Container(props: {
	title: string
	href: string
	desc: string
}): JSX.Element {
	return (
		<div class="flex h-36 flex-col items-center justify-center not-last:border-b text-center md:h-48 md:border lg:w-128">
			<h2 class="text-xl lg:text-3xl">
				<a class="" href={props.href}>
					{props.title}
				</a>
			</h2>
			<p class="mt-4 lg:text-xl">{props.desc}</p>
		</div>
	)
}
