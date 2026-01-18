import { createContext, createEffect, Index, onMount } from "solid-js"
import { createStore, type SetStoreFunction } from "solid-js/store"
import { render } from "solid-js/web"
import Tree10 from "./components/Tree10"
import { getElementById } from "./utils"

import "./index.css"

export const [state, setState] = createStore({
	kernelRowStart: 0,
	kernelRowEnd: 0,
	currentRow: -1,

	kernelName: "StatesStates",

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
			<Elements rows={5} columns={10} />
			<ActiveKernel />
		</div>
	)
}

function RowsInfo() {
	return (
		<div class="mr-[10px] flex items-center">
			<span class="-mr-[15px] inline-block -rotate-90">Nodes</span>
			<div class="flex flex-col space-y-[6px] text-right text-[16px] leading-none">
				<div>6</div>
				<div>7</div>
				<div>8</div>
				<div>9</div>
				<div>10</div>
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
				{(_, index) => <Row currentRow={index} numColumns={props.columns} />}
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

function Row(props: { currentRow: number; numColumns: number }) {
	return (
		<div class="flex space-x-[6px]">
			<Index each={Array(props.numColumns)}>
				{() => <Block currentRow={props.currentRow} />}
			</Index>
		</div>
	)
}

function Block(props: { currentRow: number }) {
	const color = () =>
		state.currentRow === props.currentRow ? "bg-black" : "bg-white"

	return <div class={`size-[16px] border ${color()} transition-colors`}></div>
}

const NAMES = [
	"StatesStates",
	"StatesStates",
	"StatesPartials",
	"StatesPartials",
	"PartialsPartials",
]

function prev() {
	if (state.currentRow === -1) {
		return
	}
	if (state.currentRow === 0) {
		setState("currentRow", -1)
		return
	}

	setState("kernelRowStart", (start) => start - 1)
	setState("kernelRowEnd", (end) => end - 1)
	setState("currentRow", (row) => row - 1)
}

function next() {
	if (state.currentRow === -1) {
		setState("currentRow", (row) => row + 1)
		return
	}

	if (state.currentRow === 4) {
		return
	}

	setState("kernelRowStart", (start) => start + 1)
	setState("kernelRowEnd", (end) => end + 1)
	setState("currentRow", (row) => row + 1)
}

createEffect(() => {
	setState("kernelName", NAMES[Math.max(0, state.currentRow)])
})

createEffect(() => {
	setState("selectedNodes", Array(11).fill(false))
	setState("selectedEdges", Array(10).fill(false))
	if (state.currentRow > -1) {
		setState("selectedNodes", state.currentRow + 6, true)
		setState("selectedEdges", state.currentRow * 2, true)
		setState("selectedEdges", state.currentRow * 2 + 1, true)
	}
})

render(() => <Main />, getElementById("kernelsBeast"))
