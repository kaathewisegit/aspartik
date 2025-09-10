import Html from "../components/html"

export default function () {
	return (
		<Html title="Home">
			<section class="flex min-h-screen min-w-screen flex-col items-center justify-center">
				<h1 class="mx-auto max-w-fit text-5xl font-medium lg:text-9xl">
					Aspartik b3
				</h1>
				<p class="mt-8 text-center text-xl font-light lg:text-4xl">
					<span class="casl italic">Fast</span> and{" "}
					<span class="casl italic">efficient</span> Bayesian phylogenetic
					analysis
				</p>

				<nav class="absolute right-0 bottom-0 m-4 flex flex-col space-y-2 text-right lg:m-16 lg:space-y-4 lg:text-3xl">
					<a href="/docs/">Documentation</a>
					<a href="https://github.com/kaathewisegit/aspartik/">Source code</a>
				</nav>
			</section>
		</Html>
	)
}
