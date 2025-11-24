import Html from "../../components/html"

export default function () {
	return (
		<Html
			title="Documentation"
			class="flex min-h-screen items-center justify-center"
		>
			<div class="flex w-full max-w-5xl flex-col gap-y-4 md:grid md:grid-cols-2 md:gap-x-8">
				<Container
					title="Tutorials"
					href="./tutorials"
					desc="Step by step lessons"
				/>
				<Container
					title="How-to guides"
					href="./howto"
					desc="Solutions to concrete tasks"
				/>
				<Container
					title="Explanation"
					href="./explanation"
					desc="Architecture and theory notes"
				/>
				<Container
					title="Reference"
					href="./reference"
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
		<div class="flex h-48 flex-col items-center justify-center border text-center lg:w-128">
			<h2 class="text-xl lg:text-3xl">
				<a class="" href={props.href}>
					{props.title}
				</a>
			</h2>
			<p class="mt-4 lg:text-xl">{props.desc}</p>
		</div>
	)
}
