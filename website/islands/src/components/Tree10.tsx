import { Index } from "solid-js"
import type { Store } from "solid-js/store"

const NODES = [
	{ x: 0, y: 100 },
	{ x: 20, y: 100 },
	{ x: 40, y: 100 },
	{ x: 60, y: 100 },
	{ x: 80, y: 100 },
	{ x: 100, y: 100 },
	{ x: 10, y: 65 },
	{ x: 70, y: 75 },
	{ x: 25, y: 40 },
	{ x: 90, y: 56 },
	{ x: 50, y: 20 },
]

const LINES = [
	[0, 6],
	[1, 6],
	[2, 8],
	[3, 7],
	[4, 7],
	[5, 9],
	[6, 8],
	[7, 9],
	[8, 10],
	[9, 10],
]

export default function Tree10(props: {
	selectedNodes: Store<boolean[]>
	selectedEdges: Store<boolean[]>
}) {
	return (
		<svg viewBox="-10 10 120 100" xmlns="http://www.w3.org/2000/svg">
			<title>A phylogenetic tree with 10 nodes</title>
			<Index each={LINES}>
				{(value, index) => (
					<Line
						from={NODES[value()[0]]}
						to={NODES[value()[1]]}
						selected={props.selectedEdges[index]}
					/>
				)}
			</Index>
			<Index each={NODES}>
				{(node, index) => (
					<Node
						index={index}
						selected={props.selectedNodes[index]}
						{...node()}
					/>
				)}
			</Index>
		</svg>
	)
}

function Node(props: {
	x: number
	y: number
	index: number
	selected: boolean
}) {
	const fill = () => (props.selected ? "#000" : "#fff")
	const fillR = () => (props.selected ? "#fff" : "#000")

	return (
		<>
			<circle
				class="transition-all"
				cx={props.x}
				cy={props.y}
				r="6"
				stroke="#333"
				stroke-width="0.5"
				fill={fill()}
			/>
			<text
				class="transition-all"
				x={props.x}
				y={props.y}
				font-size="8"
				text-anchor="middle"
				dominant-baseline="central"
				fill={fillR()}
			>
				{props.index}
			</text>
		</>
	)
}

function Line(props: {
	from: { x: number; y: number }
	to: { x: number; y: number }
	selected: boolean
}) {
	const width = () => (props.selected ? "1.5" : "0.5")

	return (
		<line
			class="transition-all"
			x1={props.from.x}
			y1={props.from.y}
			x2={props.to.x}
			y2={props.to.y}
			stroke="#333"
			stroke-width={width()}
		/>
	)
}
