import Html from "../../components/html"

export default function () {
	return (
		<Html class="flex min-h-screen items-center justify-center">
			<div class="flex w-full max-w-5xl flex-col gap-y-4 md:grid md:grid-cols-2 md:gap-x-8">
				<Container title="Tutorials" href="./tutorials" />
				<Container title="How-to guides" href="./howto" />
				<Container title="Explanation" href="./explanation" />
				<Container title="Reference" href="./reference" />
			</div>
		</Html>
	)
}

export function Container(props: { title: string; href: string }): JSX.Element {
	return (
		<div class="flex h-48 items-center justify-center border lg:w-128">
			<h2 class="text-xl lg:text-3xl">
				<a href={props.href}>{props.title}</a>
			</h2>
		</div>
	)
}
