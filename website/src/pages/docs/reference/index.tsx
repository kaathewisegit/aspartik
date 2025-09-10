import Html from "../../../components/html"

const REDIRECT = `window.location.href = "/docs/reference/aspartik"`

export default function () {
	return (
		<Html title="Reference">
			<script>{REDIRECT}</script>
		</Html>
	)
}
