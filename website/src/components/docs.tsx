export function Container(props: { title: string; href: string }): JSX.Element {
	return (
		<div class="flex h-48 items-center justify-center border lg:w-128">
			<h2 class="text-xl lg:text-3xl">
				<a href={props.href}>{props.title}</a>
			</h2>
		</div>
	)
}

// TODO: moving away from Astro
export const CONTAINERS = (
	<>
		<Container title="Tutorials" href="./tutorials" />
		<Container title="How-to guides" href="./howto" />
		<Container title="Explanation" href="./explanation" />
		<Container title="Reference" href="./reference" />
	</>
)
