import "../index.css"

import { createContext, createEffect, Index } from "solid-js"
import { createStore, type SetStoreFunction } from "solid-js/store"

const [state, setState] = createStore({
	kernelRowStart: 1,
	kernelRowEnd: 3,
	currentRow: 2,
})

export const Context = createContext<{
	state: typeof state
	setState: SetStoreFunction<typeof state>
}>({ state, setState })

export default function Main() {
	setTimeout(() => {
		setState("kernelRowStart", 2)
		console.log(state)
	}, 1000)
	setState("kernelRowStart", 1)

	return <Visualization />
}

function Visualization() {
	return (
		<div class="flex w-fit p-4">
			<Elements rows={7} columns={10} />
		</div>
	)
}

function Elements(props: { rows: number; columns: number }) {
	return (
		<div class="flex flex-col space-y-[10px]">
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
		const startPx = state.kernelRowStart * 35 - 4
		const num = state.kernelRowEnd - state.kernelRowStart
		const widthPx = (num + 1) * 25 + num * 10 + 8

		element.style.transform = `translate(-8px, ${startPx}px)`
		element.style.height = `${widthPx}px`
	})

	return (
		<div
			ref={element}
			class="absolute -z-10 w-[356px] bg-gray-400 transition-all"
		></div>
	)
}

function Row(props: { currentRow: number; numColumns: number }) {
	return (
		<div class="flex space-x-[10px]">
			<Index each={Array(props.numColumns)}>
				{() => <Block currentRow={props.currentRow} />}
			</Index>
		</div>
	)
}

function Block(props: { currentRow: number }) {
	const color = () =>
		state.currentRow === props.currentRow ? "bg-black" : "bg-white"

	return <div class={`size-[25px] border ${color()}`}></div>
}
