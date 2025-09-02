export default {
	tabWidth: 8,
	useTabs: true,
	semi: false,
	endOfLine: "auto",

	plugins: [
		"prettier-plugin-astro",
		// must be last
		// https://github.com/tailwindlabs/prettier-plugin-tailwindcss#compatibility-with-other-prettier-plugins
		"prettier-plugin-tailwindcss",
	],

	overrides: [
		{
			files: "*.astro",
			options: {
				parser: "astro",
			},
		},
	],
}
