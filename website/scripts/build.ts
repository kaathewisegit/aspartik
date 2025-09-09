import { build } from "@kaathewise/ssg"
import { $ } from "bun"

await build("src/pages/", "../target/website/")
await $`cp -r public/* ../target/website/`
