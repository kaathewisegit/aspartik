import type { PropsWithChildren } from "@kitajs/html"

export default function (props: PropsWithChildren<{ class?: string }>) {
	return (
		<>
			{"<!DOCTYPE html>"}
			<html lang="en">
				<head>
					<meta charset="UTF-8" />
					<meta name="viewport" content="width=device-width" />
					<link href="/style.css" rel="stylesheet" />
					<title>Aspartik</title>
				</head>
				<body class={props.class}>{props.children}</body>
			</html>
		</>
	)
}
