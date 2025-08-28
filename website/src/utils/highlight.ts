import "./highlight.css"
import { parser } from "@lezer/python"
import { highlightCode, classHighlighter } from "@lezer/highlight"

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

	highlightCode(
		code,
		parser.parse(code),
		classHighlighter,
		emit,
		emitBreak,
	)

	return result
}
