import { createContext, createEffect, Index, onMount } from "solid-js"
import { createStore, type SetStoreFunction } from "solid-js/store"
import { render } from "solid-js/web"
import Tree10 from "./components/Tree10"
import { getElementById } from "./utils"

import "./index.css"

export const [state, setState] = createStore({
	kernelRowStart: 0,
	kernelRowEnd: 0,
	currentStep: -1,

	kernelName: "update_leaves",

	selectedNodes: Array(11).fill(false) as boolean[],
	selectedEdges: Array(10).fill(false) as boolean[],
})

export const Context = createContext<{
	state: typeof state
	setState: SetStoreFunction<typeof state>
}>({ state, setState })

export default function Main() {
	const handler = (e: KeyboardEvent) => {
		if (e.key === "ArrowLeft") {
			prev()
		} else if (e.key === "ArrowRight") {
			next()
		}
	}

	onMount(() => window.addEventListener("keydown", handler))

	return (
		<div class="flex flex-col items-center [&_*]:duration-300">
			<figure class="h-54 w-64">
				<Tree10
					selectedNodes={state.selectedNodes}
					selectedEdges={state.selectedEdges}
				/>
			</figure>
			<Visualization />
			<Controls />
		</div>
	)
}

function Visualization() {
	return (
		<div class="m-4 flex w-fit">
			<RowsInfo />
			<Elements rows={10} columns={10} />
			<ActiveKernel />
		</div>
	)
}

function RowsInfo() {
	return (
		<div class="mr-[10px] flex items-center">
			<span class="-mr-[5px] inline-block -rotate-90">Edges</span>
			<div class="flex flex-col space-y-[6px] text-right text-[16px] leading-none">
				<div>0</div>
				<div>1</div>
				<div>2</div>
				<div>3</div>
				<div>4</div>
				<div>5</div>
				<div>6</div>
				<div>7</div>
				<div>8</div>
				<div>9</div>
			</div>
		</div>
	)
}

function ActiveKernel() {
	let element!: HTMLSpanElement

	createEffect(() => {
		const pos = (state.kernelRowStart + state.kernelRowEnd) / 2
		const posPx = pos * 22
		element.style.transform = `translate(0px, ${posPx}px)`
	})

	return (
		<span
			ref={element}
			class="ml-[10px] w-[80px] text-[16px] leading-none transition-all"
		>
			<code>{state.kernelName}</code>
		</span>
	)
}

function Controls() {
	return (
		<div class="m-2 flex w-fit space-x-6">
			<button class="border px-2" type="button" on:click={prev}>
				← Previous
			</button>
			<button class="border px-2" type="button" on:click={next}>
				Next →
			</button>
		</div>
	)
}

function Elements(props: { rows: number; columns: number }) {
	return (
		<div class="flex flex-col space-y-[6px]">
			<Active />

			<Index each={Array(props.rows)}>
				{(_, index) => <Row edge={index} numColumns={props.columns} />}
			</Index>
		</div>
	)
}

function Active() {
	let element!: HTMLDivElement

	createEffect(() => {
		const startPx = state.kernelRowStart * 22 - 3
		const num = state.kernelRowEnd - state.kernelRowStart
		const widthPx = (num + 1) * 16 + num * 6 + 6

		element.style.transform = `translate(-5px, ${startPx}px)`
		element.style.height = `${widthPx}px`
	})

	return (
		<div
			ref={element}
			class="absolute -z-10 w-[224px] bg-gray-400 transition-all"
		></div>
	)
}

function Row(props: { edge: number; numColumns: number }) {
	return (
		<div class="flex space-x-[6px]">
			<Index each={Array(props.numColumns)}>
				{() => <Block edge={props.edge} />}
			</Index>
		</div>
	)
}

function Block(props: { edge: number }) {
	const color = () =>
		state.selectedEdges[props.edge] ? "bg-black" : "bg-white"

	return <div class={`size-[16px] border ${color()} transition-colors`}></div>
}

function prev() {
	if (state.currentStep === -1) {
		return
	} else {
		setState("currentStep", (step) => step - 1)
	}
}

function next() {
	if (state.currentStep === 4) {
		return
	} else {
		setState("currentStep", (step) => step + 1)
	}
}

createEffect(() => {
	if (state.currentStep <= 0) {
		setState("kernelName", "update_leaves")
	} else {
		setState("kernelName", "propose")
	}
})

createEffect(() => {
	if (state.currentStep <= 0) {
		setState("kernelRowStart", 0)
		setState("kernelRowEnd", 5)
	} else {
		setState("kernelRowStart", 6)
		setState("kernelRowEnd", 9)
	}
})

createEffect(() => {
	setState("selectedEdges", Array(10).fill(false))
	if (state.currentStep === 0) {
		setState("selectedEdges", (_, index) => index < 6, true)
	} else if (state.currentStep > 0) {
		setState("selectedEdges", state.currentStep + 5, true)
	}
})

render(() => <Main />, getElementById("kernelsB3"))
