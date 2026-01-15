import Html from "../../components/html.tsx"

const REDIRECT = `window.location.href = "/reference/aspartik"`

export default function () {
	return (
		<Html title="Reference">
			<script>{REDIRECT}</script>
		</Html>
	)
}
