import { defineCollection } from "astro:content"
import fs from "node:fs/promises"
import {
	type ClassType,
	type FunctionType,
	type ModuleType,
	moduleSchema,
	type VariableType,
} from "./schema"
import { md2html } from "./utils"

async function renderDocstrings(
	value: ModuleType | ClassType | FunctionType | VariableType,
): Promise<void> {
	if (value.docstring) {
		value.docstring = await md2html(value.docstring)
	}

	if (value.type === "module") {
		value.classes.forEach(renderDocstrings)
		value.variables.forEach(renderDocstrings)
		value.functions.forEach(renderDocstrings)
	}

	if (value.type === "class") {
		value.instance_variables.forEach(renderDocstrings)
		value.class_variables.forEach(renderDocstrings)
		value.classmethods.forEach(renderDocstrings)
		value.staticmethods.forEach(renderDocstrings)
		value.methods.forEach(renderDocstrings)
	}
}

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
		renderDocstrings(module)

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
