export function getElementById(id: string): HTMLElement {
	const el = document.getElementById(id)

	if (!(el instanceof HTMLElement)) {
		throw new Error(`Element with id ${id} not found`)
	}

	return el
}
