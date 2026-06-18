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
	const currentStep = signal(-1)
	const kernelRowStart = computed(() => (currentStep.value <= 0 ? 0 : 6))
	const kernelRowEnd = computed(() => (currentStep.value <= 0 ? 5 : 9))
	const kernelName = computed(() =>
		currentStep.value <= 0 ? "update_leaves" : "propose",
	)
	const selectedEdges: ReadonlySignal<boolean[]> = computed(() => {
		const edges = Array(10).fill(false)
		if (currentStep.value === 0) {
			return edges.map((_, i) => i < 6)
		} else if (currentStep.value > 0) {
			edges[currentStep.value + 5] = true
		}
		return edges
	})
	const selectedNodes: ReadonlySignal<boolean[]> = computed(() =>
		Array(11).fill(false),
	)

	return {
		currentStep,

		kernelRowStart,
		kernelRowEnd,

		kernelName,

		selectedEdges,
		selectedNodes,

		prev() {
			if (currentStep.value > -1) currentStep.value -= 1
		},

		next() {
			if (currentStep.value < 4) currentStep.value += 1
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
			<Elements rows={10} numColumns={10} />
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

function Elements(props: { rows: number; numColumns: number }) {
	return (
		<div class="flex flex-col space-y-[6px]">
			<Active />

			{Array.from({ length: props.rows }).map((_, i) => (
				<Row key={i} edge={i} numColumns={props.numColumns} />
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
		></div>
	)
}

function Row(props: { edge: number; numColumns: number }) {
	return (
		<div class="flex space-x-[6px]">
			{Array.from({ length: props.numColumns }).map((_, i) => (
				<Block key={i} edge={props.edge} />
			))}
		</div>
	)
}

function Block(props: { edge: number }) {
	const color = () =>
		model.selectedEdges.value[props.edge] ? "bg-black" : "bg-white"

	return <div class={`size-[16px] border ${color()} transition-colors`}></div>
}

function ActiveKernel() {
	const elementRef = useRef<HTMLSpanElement>(null)

	useSignalEffect(() => {
		const el = elementRef.current
		if (!el) return

		const pos = (model.kernelRowStart.value + model.kernelRowEnd.value) / 2
		const posPx = pos * 22
		el.style.transform = `translate(0px, ${posPx}px)`
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

render(<Main />, getElementById("kernelsB3"))
