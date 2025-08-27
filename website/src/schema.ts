import { z } from "astro/zod"

const commonKeys = z.object({
	type: z.string(),
	fullname: z.string(),
	name: z.string(),
	docstring: z.string().nullable(),
	source: z.string().nullable(),
	source_lines: z
		.tuple([
			z.number().int().nonnegative(),
			z.number().int().nonnegative(),
		])
		.nullable(),
	source_file: z.string().nullable(),
})

export const variableSchema = commonKeys
	.merge(
		z.object({
			type: z.literal("variable"),
			annotation: z.string().nullable(),
			default: z.string(),
		}),
	)
	.strict()

export type VariableType = z.infer<typeof variableSchema>

export const functionSchema = commonKeys.merge(
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

export const classSchema = commonKeys.merge(
	z.object({
		type: z.literal("class"),
		bases: z.array(z.tuple([z.string(), z.string(), z.string()])),
		class_variables: z.array(variableSchema),
		instance_variables: z.array(variableSchema),
		classmethods: z.array(functionSchema),
		staticmethods: z.array(functionSchema),
		methods: z.array(functionSchema),
	}),
)

export type ClassType = z.infer<typeof classSchema>

export const moduleSchema = commonKeys.merge(
	z.object({
		type: z.literal("module"),
		id: z.string(),
		submodules: z.array(
			z.object({
				fullname: z.string(),
				name: z.string(),
			}),
		),
		classes: z.array(classSchema),
		functions: z.array(functionSchema),
		variables: z.array(variableSchema),
	}),
)

export type ModuleType = z.infer<typeof moduleSchema>
