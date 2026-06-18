const REDIRECT = `window.location.href = "/reference/aspartik"`

export function Head() {
	return <script>{REDIRECT}</script>
}

export function Body() {
	return ""
}
