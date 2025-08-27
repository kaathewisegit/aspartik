import { z } from "astro/zod"

export const moduleSchema = z.object({
	id: z.string(),
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
	submodules: z.array(
		z.object({
			fullname: z.string(),
			name: z.string(),
		}),
	),
	classes: z.array(z.unknown()),
	functions: z.array(z.unknown()),
	variables: z.array(z.unknown()),
})

export type ModuleType = z.infer<typeof moduleSchema>
