import {
	computed,
	createModel,
	type ReadonlySignal,
	signal,
	useSignalEffect,
} from "@preact/signals"
import { render } from "preact"
import { useEffect, useRef } from "preact/hooks"
import Tree10 from "./components/Tree10"
import { getElementById } from "./utils"
import "./index.css"

const Model = createModel(() => {
	const currentRow = signal(-1)

	const kernelName = computed(
		() =>
			[
				"StatesStates",
				"StatesStates",
				"StatesPartials",
				"StatesPartials",
				"PartialsPartials",
			][Math.max(0, currentRow.value)],
	)

	const kernelRowStart = computed(() => Math.max(0, currentRow.value))
	const kernelRowEnd = kernelRowStart

	const selectedNodes: ReadonlySignal<boolean[]> = computed(() => {
		const nodes = Array(11).fill(false)
		if (currentRow.value > -1) {
			nodes[currentRow.value + 6] = true
		}
		return nodes
	})

	const selectedEdges: ReadonlySignal<boolean[]> = computed(() => {
		const edges = Array(10).fill(false)
		const row = currentRow.value
		if (row === -1) return edges

		const edgeConfig: Record<number, number[]> = {
			0: [0, 1],
			1: [3, 4],
			2: [6, 2],
			3: [7, 5],
			4: [8, 9],
		}

		edgeConfig[row]?.forEach((idx) => {
			edges[idx] = true
		})

		return edges
	})

	return {
		currentStep: currentRow,

		kernelRowStart,
		kernelRowEnd,

		kernelName,

		selectedEdges,
		selectedNodes,

		prev() {
			if (currentRow.value > -1) currentRow.value -= 1
		},

		next() {
			if (currentRow.value < 4) currentRow.value += 1
		},
	}
})
const model = new Model()

export default function Main() {
	useEffect(() => {
		const handler = (e: KeyboardEvent) => {
			if (e.key === "ArrowLeft") model.prev()
			else if (e.key === "ArrowRight") model.next()
		}
		window.addEventListener("keydown", handler)
		return () => window.removeEventListener("keydown", handler)
	}, [])

	return (
		<div class="flex flex-col items-center [&_*]:duration-300">
			<figure class="h-54 w-64">
				<Tree10
					selectedNodes={model.selectedNodes}
					selectedEdges={model.selectedEdges}
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

function Elements(props: { rows: number; columns: number }) {
	return (
		<div class="relative flex flex-col space-y-[6px]">
			<Active />
			{Array.from({ length: props.rows }).map((_, i) => (
				<Row key={i} currentRow={i} numColumns={props.columns} />
			))}
		</div>
	)
}

function Active() {
	const elementRef = useRef<HTMLDivElement>(null)

	useSignalEffect(() => {
		const el = elementRef.current
		if (!el) return

		const startPx = model.kernelRowStart.value * 22 - 3
		const num = model.kernelRowEnd.value - model.kernelRowStart.value
		const widthPx = (num + 1) * 16 + num * 6 + 6

		el.style.transform = `translate(-5px, ${startPx}px)`
		el.style.height = `${widthPx}px`
	})

	return (
		<div
			ref={elementRef}
			class="absolute -z-10 w-[224px] bg-gray-400 transition-all"
		/>
	)
}

function Row(props: { currentRow: number; numColumns: number }) {
	return (
		<div class="flex space-x-[6px]">
			{Array.from({ length: props.numColumns }).map((_, i) => (
				<Block key={i} currentRow={props.currentRow} />
			))}
		</div>
	)
}

function Block(props: { currentRow: number }) {
	const color = computed(() =>
		model.currentStep.value === props.currentRow ? "bg-black" : "bg-white",
	)
	return <div class={`size-[16px] border ${color.value} transition-colors`} />
}

function ActiveKernel() {
	const elementRef = useRef<HTMLSpanElement>(null)

	useSignalEffect(() => {
		if (!elementRef.current) return
		const pos = (model.kernelRowStart.value + model.kernelRowEnd.value) / 2
		elementRef.current.style.transform = `translate(0px, ${pos * 22}px)`
	})

	return (
		<span
			ref={elementRef}
			class="ml-[10px] w-[80px] text-[16px] leading-none transition-all"
		>
			<code>{model.kernelName}</code>
		</span>
	)
}

function Controls() {
	return (
		<div class="m-2 flex w-fit space-x-6">
			<button class="border px-2" type="button" onClick={model.prev}>
				← Previous
			</button>
			<button class="border px-2" type="button" onClick={model.next}>
				Next →
			</button>
		</div>
	)
}

render(<Main />, getElementById("kernelsBeast"))
