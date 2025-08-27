import { defineCollection } from "astro:content"
import fs from "node:fs/promises"
import { type ModuleType, moduleSchema } from "./schema"
import { md2html } from "./utils"

async function flattenModules(
	module: Record<string, any>,
): Promise<ModuleType[]> {
	const modules = [] as any[]

	async function traverse(module: any) {
		const submodules = module.submodules

		module.id = module.fullname
		module.submodules = module.submodules.map((mod: any) => ({
			fullname: mod.fullname,
			name: mod.name,
		}))
		if (module.docstring) {
			module.docstring = await md2html(module.docstring)
		}

		modules.push(module)

		submodules.forEach(traverse)
	}

	traverse(module)

	return modules
}

const reference = defineCollection({
	loader: async () => {
		const text = await fs.readFile("../target/pdoc.json", "utf8")
		const json = JSON.parse(text)
		const modules = await flattenModules(json)
		return modules
	},
	schema: moduleSchema,
})

export const collections = { reference }
