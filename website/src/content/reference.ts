import fs from "node:fs/promises"
import { z } from "zod"

const commonKeys = z.object({
	type: z.string(),
	qualname: z.string(),
	fullname: z.string(),
	name: z.string(),
	docstring: z.string().nullable(),
	source: z.string().nullable(),
	source_lines: z
		.tuple([z.number().int().nonnegative(), z.number().int().nonnegative()])
		.nullable(),
	source_file: z.string().nullable(),
})

export type CommonType = z.infer<typeof commonKeys>

export const variableSchema = commonKeys.and(
	z.object({
		type: z.literal("variable"),
		annotation: z.string().nullable(),
		default: z.string(),
	}),
)

export type VariableType = z.infer<typeof variableSchema>

export const functionSchema = commonKeys.and(
	z.object({
		type: z.literal("function"),
		classmethod: z.boolean(),
		staticmethod: z.boolean(),
		decorators: z.array(z.string()),
		def: z.enum(["def", "async def"]),
		signature: z.string(),
		signature_without_self: z.string(),
	}),
)

export type FunctionType = z.infer<typeof functionSchema>

export const classSchema = commonKeys.and(
	z.object({
		type: z.literal("class"),
		decorators: z.array(z.string()),
		bases: z.array(z.tuple([z.string(), z.string(), z.string()])),
		class_variables: z.array(variableSchema),
		instance_variables: z.array(variableSchema),
		classmethods: z.array(functionSchema),
		staticmethods: z.array(functionSchema),
		methods: z.array(functionSchema),
	}),
)

export type ClassType = z.infer<typeof classSchema>

export const SubmoduleSchema = z.object({
	fullname: z.string(),
	name: z.string(),
})

export type SubmoduleType = z.infer<typeof SubmoduleSchema>

export const moduleSchema = commonKeys.and(
	z.object({
		type: z.literal("module"),
		id: z.string(),
		submodules: z.array(SubmoduleSchema),
		classes: z.array(classSchema),
		functions: z.array(functionSchema),
		variables: z.array(variableSchema),
	}),
)

export type ModuleType = z.infer<typeof moduleSchema>

async function renderDocstrings(
	value: ModuleType | ClassType | FunctionType | VariableType,
): Promise<void> {
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

const text = await fs.readFile("../target/pdoc.json", "utf8")
const json = JSON.parse(text)
const modules = (await flattenModules(json)) as ModuleType[]

export default modules
