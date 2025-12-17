import { classHighlighter, highlightCode } from "@lezer/highlight"
import { parser } from "@lezer/python"

export default function highlight(code: string, _lang: "python"): string {
	let result = ""

	function emit(text: string, classes: string) {
		if (classes) {
			result += `<span class="${classes}">${text}</span>`
		} else {
			result += text
		}
	}
	function emitBreak() {
		result += "\n"
	}

	highlightCode(code, parser.parse(code), classHighlighter, emit, emitBreak)

	return result
}
