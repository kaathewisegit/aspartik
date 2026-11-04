use anyhow::{Result, ensure};

use std::fmt::{self, Write};

use super::{BinaryTree, Node, RootedTree};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
	pub x: f64,
	pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutKind {
	Rectangular,
	Slanted,
	Tidy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TreeLayout {
	coordinates: Box<[Point]>,
	width: f64,
	height: f64,
	kind: LayoutKind,
}

impl TreeLayout {
	pub fn point(&self, node: Node) -> Option<Point> {
		self.coordinates.get(node.usize()).copied()
	}

	pub fn points(&self) -> &[Point] {
		&self.coordinates
	}

	pub fn width(&self) -> f64 {
		self.width
	}

	pub fn height(&self) -> f64 {
		self.height
	}

	pub fn kind(&self) -> LayoutKind {
		self.kind
	}
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvgOptions {
	pub x_scale: f64,
	pub y_scale: f64,
	pub margin: f64,
	pub node_radius: f64,
	pub font_size: f64,
	pub show_names: bool,
}

impl Default for SvgOptions {
	fn default() -> Self {
		Self {
			x_scale: 100.0,
			y_scale: 30.0,
			margin: 20.0,
			node_radius: 3.0,
			font_size: 12.0,
			show_names: true,
		}
	}
}

#[derive(Debug, Clone, Copy)]
struct TidyNode {
	preliminary: f64,
	modifier: f64,
	left_thread: Option<Node>,
	right_thread: Option<Node>,
	extreme_left: Node,
	extreme_right: Node,
	modifier_extreme_left: f64,
	modifier_extreme_right: f64,
}

impl BinaryTree {
	pub fn rectangular_layout(
		&self,
		separation: f64,
	) -> Result<TreeLayout> {
		self.phylogram_layout(separation, LayoutKind::Rectangular)
	}

	pub fn slanted_layout(&self, separation: f64) -> Result<TreeLayout> {
		self.phylogram_layout(separation, LayoutKind::Slanted)
	}

	fn phylogram_layout(
		&self,
		separation: f64,
		kind: LayoutKind,
	) -> Result<TreeLayout> {
		let distances = self.layout_distances(separation)?;
		let mut coordinates = vec![
			Point { x: 0.0, y: 0.0 };
			self.num_nodes() as usize
		];
		let mut next_leaf = 0_u32;
		for node in self.preorder() {
			coordinates[node.usize()].x = distances[node.usize()];
			if self.is_leaf(node) {
				coordinates[node.usize()].y =
					f64::from(next_leaf) * separation;
				next_leaf += 1;
			}
		}
		for node in self.postorder() {
			let Some(internal) = self.as_internal(node) else {
				continue;
			};
			let [left, right] = self.children_of(internal);
			coordinates[node.usize()].y =
				(coordinates[left.usize()].y
					+ coordinates[right.usize()].y) / 2.0;
		}
		Ok(TreeLayout {
			width: distances.into_iter().fold(0.0, f64::max),
			height: f64::from(next_leaf.saturating_sub(1))
				* separation,
			coordinates: coordinates.into_boxed_slice(),
			kind,
		})
	}

	pub fn tidy_layout(&self, separation: f64) -> Result<TreeLayout> {
		let distances = self.layout_distances(separation)?;
		let mut state = self
			.nodes()
			.map(|node| TidyNode {
				preliminary: 0.0,
				modifier: 0.0,
				left_thread: None,
				right_thread: None,
				extreme_left: node,
				extreme_right: node,
				modifier_extreme_left: 0.0,
				modifier_extreme_right: 0.0,
			})
			.collect::<Vec<_>>();

		for node in self.postorder() {
			let Some(internal) = self.as_internal(node) else {
				continue;
			};
			let [left, right] = self.children_of(internal);
			separate_subtrees(
				self, left, right, &distances, &mut state,
				separation,
			);
			state[node.usize()].preliminary = (state[left.usize()]
				.preliminary
				+ state[left.usize()].modifier
				+ state[right.usize()].modifier
				+ state[right.usize()].preliminary)
				/ 2.0;
			state[node.usize()].extreme_left =
				state[left.usize()].extreme_left;
			state[node.usize()].modifier_extreme_left =
				state[left.usize()].modifier_extreme_left;
			state[node.usize()].extreme_right =
				state[right.usize()].extreme_right;
			state[node.usize()].modifier_extreme_right =
				state[right.usize()].modifier_extreme_right;
		}

		let mut vertical = vec![0.0; self.num_nodes() as usize];
		let mut minimum = f64::INFINITY;
		let mut maximum = f64::NEG_INFINITY;
		let mut stack = vec![(Node::from(self.root()), 0.0)];
		while let Some((node, modifier_sum)) = stack.pop() {
			let item = state[node.usize()];
			let modifier_sum = modifier_sum + item.modifier;
			let value = item.preliminary + modifier_sum;
			vertical[node.usize()] = value;
			minimum = minimum.min(value);
			maximum = maximum.max(value);
			if let Some(internal) = self.as_internal(node) {
				let [left, right] = self.children_of(internal);
				stack.push((right, modifier_sum));
				stack.push((left, modifier_sum));
			}
		}

		let coordinates = self
			.nodes()
			.map(|node| Point {
				x: distances[node.usize()],
				y: vertical[node.usize()] - minimum,
			})
			.collect::<Vec<_>>();
		Ok(TreeLayout {
			width: distances.into_iter().fold(0.0, f64::max),
			height: maximum - minimum,
			coordinates: coordinates.into_boxed_slice(),
			kind: LayoutKind::Tidy,
		})
	}

	pub fn to_svg<NC, EC, N, E>(
		&self,
		layout: &TreeLayout,
		options: SvgOptions,
		node_color: NC,
		edge_color: EC,
	) -> Result<String>
	where
		NC: Fn(Node) -> N,
		EC: Fn(Node) -> E,
		N: AsRef<str>,
		E: AsRef<str>,
	{
		ensure!(
			layout.coordinates.len() == self.num_nodes() as usize,
			"The layout does not match the tree"
		);
		validate_svg_options(options)?;

		let scaled = |point: Point| Point {
			x: options.margin + point.x * options.x_scale,
			y: options.margin + point.y * options.y_scale,
		};
		let label_gap = options.node_radius + options.font_size * 0.5;
		let mut content_width = layout.width * options.x_scale;
		if options.show_names {
			for leaf in self.leaves() {
				let node = Node::from(leaf);
				let Some(name) = self.name(node) else {
					continue;
				};
				let x = layout.coordinates[node.usize()].x
					* options.x_scale;
				content_width = content_width.max(x
					+ label_gap
					+ options.font_size
						* 0.6 * name.chars().count() as f64);
			}
		}
		let width = content_width + options.margin * 2.0;
		let height =
			layout.height * options.y_scale + options.margin * 2.0;
		let mut output = String::new();
		output.push_str(
			"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"",
		);
		write_horizontal(&mut output, width)?;
		write!(output, "\" height=\"{height}\" viewBox=\"0 0 ")?;
		write_horizontal(&mut output, width)?;
		write!(output, " {height}\">")?;

		for child in self.edges() {
			let parent = self.parent_of(child).unwrap();
			let parent_point =
				scaled(layout.coordinates[parent.usize()]);
			let child_point =
				scaled(layout.coordinates[child.usize()]);
			let color = edge_color(child);
			match layout.kind {
				LayoutKind::Rectangular | LayoutKind::Tidy => {
					output.push_str("<path d=\"M ");
					write_horizontal(
						&mut output,
						parent_point.x,
					)?;
					write!(
						output,
						" {} V {} H ",
						parent_point.y, child_point.y
					)?;
					write_horizontal(
						&mut output,
						child_point.x,
					)?;
					output.push_str(
						"\" fill=\"none\" stroke=\"",
					);
				}
				LayoutKind::Slanted => {
					output.push_str("<line x1=\"");
					write_horizontal(
						&mut output,
						parent_point.x,
					)?;
					write!(
						output,
						"\" y1=\"{}\" x2=\"",
						parent_point.y
					)?;
					write_horizontal(
						&mut output,
						child_point.x,
					)?;
					write!(
						output,
						"\" y2=\"{}\" stroke=\"",
						child_point.y
					)?;
				}
			}
			write_escaped(&mut output, color.as_ref())?;
			output.push_str("\">");
			match layout.kind {
				LayoutKind::Rectangular | LayoutKind::Tidy => {
					output.push_str("</path>")
				}
				LayoutKind::Slanted => {
					output.push_str("</line>")
				}
			}
		}

		for node in self.nodes() {
			let point = scaled(layout.coordinates[node.usize()]);
			let color = node_color(node);
			output.push_str("<circle cx=\"");
			write_horizontal(&mut output, point.x)?;
			write!(
				output,
				"\" cy=\"{}\" r=\"{}\" fill=\"",
				point.y, options.node_radius
			)?;
			write_escaped(&mut output, color.as_ref())?;
			output.push_str("\">");
			if options.show_names && self.name(node).is_some() {
				output.push_str("<title>");
				if options.show_names
					&& let Some(name) = self.name(node)
				{
					write_escaped(&mut output, name)?;
				}
				output.push_str("</title>");
			}
			output.push_str("</circle>");
		}

		if options.show_names {
			write!(
				output,
				"<g font-size=\"{}\" dominant-baseline=\"middle\">",
				options.font_size,
			)?;
			for leaf in self.leaves() {
				let node = Node::from(leaf);
				let Some(name) = self.name(node) else {
					continue;
				};
				let point =
					scaled(layout.coordinates
						[node.usize()]);
				output.push_str("<text x=\"");
				write_horizontal(
					&mut output,
					point.x + label_gap,
				)?;
				write!(output, "\" y=\"{}\">", point.y)?;
				write_escaped(&mut output, name)?;
				output.push_str("</text>");
			}
			output.push_str("</g>");
		}
		output.push_str("</svg>");
		Ok(output)
	}

	fn layout_distances(&self, separation: f64) -> Result<Vec<f64>> {
		ensure!(
			separation.is_finite() && separation > 0.0,
			"Node separation must be finite and positive"
		);
		let mut distances = vec![0.0; self.num_nodes() as usize];
		for node in self.preorder() {
			let Some(internal) = self.as_internal(node) else {
				continue;
			};
			let [left, right] = self.children_of(internal);
			for child in [left, right] {
				let length = self.edge_length(child).unwrap();
				ensure!(
					length.is_finite() && length >= 0.0,
					"Branch length for node {} must be finite and nonnegative",
					child.u32()
				);
				distances[child.usize()] =
					distances[node.usize()] + length;
			}
		}
		Ok(distances)
	}
}

fn separate_subtrees(
	tree: &BinaryTree,
	left: Node,
	right: Node,
	distances: &[f64],
	state: &mut [TidyNode],
	separation: f64,
) {
	let mut upper = Some(left);
	let mut lower = Some(right);
	let mut upper_modifiers = state[left.usize()].modifier;
	let mut lower_modifiers = state[right.usize()].modifier;
	let mut first = true;

	while let (Some(upper_node), Some(lower_node)) = (upper, lower) {
		let distance = upper_modifiers
			+ state[upper_node.usize()].preliminary
			+ separation - lower_modifiers
			- state[lower_node.usize()].preliminary;
		if (first && distance < 0.0) || distance > 0.0 {
			lower_modifiers += distance;
			let item = &mut state[right.usize()];
			item.modifier += distance;
			item.modifier_extreme_left += distance;
			item.modifier_extreme_right += distance;
			first = false;
		}

		let upper_depth = distances[upper_node.usize()];
		let lower_depth = distances[lower_node.usize()];
		if upper_depth <= lower_depth {
			upper = next_right_contour(tree, state, upper_node);
			if let Some(node) = upper {
				upper_modifiers += state[node.usize()].modifier;
			}
		}
		if upper_depth >= lower_depth {
			lower = next_left_contour(tree, state, lower_node);
			if let Some(node) = lower {
				lower_modifiers += state[node.usize()].modifier;
			}
		}
	}

	match (upper, lower) {
		(None, Some(lower_node)) => {
			let extreme = state[left.usize()].extreme_left;
			state[extreme.usize()].left_thread = Some(lower_node);
			let difference = lower_modifiers
				- state[lower_node.usize()].modifier
				- state[left.usize()].modifier_extreme_left;
			state[extreme.usize()].modifier += difference;
			state[extreme.usize()].preliminary -= difference;
			state[left.usize()].extreme_left =
				state[right.usize()].extreme_left;
			state[left.usize()].modifier_extreme_left =
				state[right.usize()].modifier_extreme_left;
		}
		(Some(upper_node), None) => {
			let extreme = state[right.usize()].extreme_right;
			state[extreme.usize()].right_thread = Some(upper_node);
			let difference = upper_modifiers
				- state[upper_node.usize()].modifier
				- state[right.usize()].modifier_extreme_right;
			state[extreme.usize()].modifier += difference;
			state[extreme.usize()].preliminary -= difference;
			state[right.usize()].extreme_right =
				state[left.usize()].extreme_right;
			state[right.usize()].modifier_extreme_right =
				state[left.usize()].modifier_extreme_right;
		}
		_ => {}
	}
}

fn next_left_contour(
	tree: &BinaryTree,
	state: &[TidyNode],
	node: Node,
) -> Option<Node> {
	if let Some(internal) = tree.as_internal(node) {
		Some(tree.children_of(internal)[0])
	} else {
		state[node.usize()].left_thread
	}
}

fn next_right_contour(
	tree: &BinaryTree,
	state: &[TidyNode],
	node: Node,
) -> Option<Node> {
	if let Some(internal) = tree.as_internal(node) {
		Some(tree.children_of(internal)[1])
	} else {
		state[node.usize()].right_thread
	}
}

fn validate_svg_options(options: SvgOptions) -> Result<()> {
	for (name, value) in [
		("Horizontal scale", options.x_scale),
		("Vertical scale", options.y_scale),
		("Node radius", options.node_radius),
		("Font size", options.font_size),
	] {
		ensure!(
			value.is_finite() && value > 0.0,
			"{name} must be finite and positive"
		);
	}
	ensure!(
		options.margin.is_finite() && options.margin >= 0.0,
		"Margin must be finite and nonnegative"
	);
	Ok(())
}

fn write_escaped(writer: &mut impl fmt::Write, value: &str) -> fmt::Result {
	for character in value.chars() {
		match character {
			'&' => writer.write_str("&amp;")?,
			'<' => writer.write_str("&lt;")?,
			'>' => writer.write_str("&gt;")?,
			'\"' => writer.write_str("&quot;")?,
			'\'' => writer.write_str("&apos;")?,
			_ => writer.write_char(character)?,
		}
	}
	Ok(())
}

fn write_horizontal(output: &mut String, value: f64) -> fmt::Result {
	let start = output.len();
	write!(output, "{value:.2}")?;
	while output.len() > start && output.ends_with('0') {
		output.pop();
	}
	if output.len() > start && output.ends_with('.') {
		output.pop();
	}
	Ok(())
}
