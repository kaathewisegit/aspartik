import Html from "../../components/html"
import Markdown from "../../components/markdown"

const MD = `
# H1 heading

Hello there.  This is a markdown example.  Here goes more text.  And here we
have \`inline verbatim\`.  Escaped, once again.

- Bullet
- List

1. Numbered
2. List

\`\`\`
That was a lot of escapes

pre:where(:not(.not-prose, .not-prose *)) {
	margin-top: calc(var(--spacing) * 4);
	margin-bottom: calc(var(--spacing) * 10);
}
\`\`\`

> A blockquote

> Another one

Some text

> And another

## h2

Some text.

---

some text

---

### h3

[A hyperlink](https://example.org)

Colons can be used to align columns.  **Strong** text.

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the
raw Markdown line up prettily. You can also use inline Markdown.

Markdown | Less | Pretty
--- | --- | ---
*Still* | \`renders\` | **nicely**
1 | 2 | 3

| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |

| Command | Description |
| --- | --- |
| git status | List all new or modified files |
| git diff | Show file differences that haven't been staged |

| Command | Description |
| --- | --- |
| \`git status\` | List all *new or modified* files |
| \`git diff\` | Show file differences that **haven't been** staged |

| Left-aligned | Center-aligned | Right-aligned |
| :---         |     :---:      |          ---: |
| git status   | git status     | git status    |
| git diff     | git diff       | git diff      |

| Name     | Character |
| ---      | ---       |
| Backtick | \`        |
| Pipe     | \\|       |
`

export default function () {
	return (
		<Html title="Tutorials">
			<Markdown raw={MD} class="prose mx-auto max-w-200" />
		</Html>
	)
}
