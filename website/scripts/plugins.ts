import { postcssPlugin } from "@kaathewise/ssg"
import tailwindcss from "@tailwindcss/postcss"
import postcss from "postcss"

Bun.plugin(postcssPlugin({ plugins: [tailwindcss() as postcss.Plugin] }))
