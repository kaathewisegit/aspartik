import { convertCSS } from "../utils/css"

export function getContentType() {
	return "text/css"
}

export default async function () {
	return await convertCSS("src/style.css")
}
