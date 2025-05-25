use anyhow::{anyhow, Result};
use vulkano::{
	buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
	command_buffer::{
		allocator::{
			StandardCommandBufferAllocator,
			StandardCommandBufferAllocatorCreateInfo,
		},
		AutoCommandBufferBuilder, CommandBufferUsage,
		PrimaryAutoCommandBuffer,
	},
	descriptor_set::{
		allocator::StandardDescriptorSetAllocator,
		layout::{
			DescriptorSetLayout, DescriptorSetLayoutBinding,
			DescriptorSetLayoutCreateInfo, DescriptorType,
		},
		DescriptorSet, WriteDescriptorSet,
	},
	device::{
		Device, DeviceCreateInfo, DeviceFeatures, Queue,
		QueueCreateInfo, QueueFlags,
	},
	instance::{Instance, InstanceCreateInfo},
	memory::allocator::{
		AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
	},
	pipeline::{
		compute::ComputePipelineCreateInfo,
		layout::{
			PipelineDescriptorSetLayoutCreateInfo,
			PipelineLayoutCreateFlags, PushConstantRange,
		},
		ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
		PipelineShaderStageCreateInfo,
	},
	shader::ShaderStages,
	sync::{self, GpuFuture},
	VulkanLibrary,
};

use std::{collections::BTreeMap, sync::Arc};

use super::{LikelihoodTrait, Row, Transition};

pub struct GpuLikelihood {
	// TODO: bench and see if allocators and such should be preserved here
	// buffers:
	// - one array for final likelihoods
	// - number of nodes
	device: Arc<Device>,
	queue: Arc<Queue>,

	reject_command: Arc<PrimaryAutoCommandBuffer>,
	update_nodes: Subbuffer<[u32]>,
	update_nodes_length: Subbuffer<u32>,

	propose_command: Arc<PrimaryAutoCommandBuffer>,
	propose_transitions: Subbuffer<[Transition<4>]>,
	propose_children: Subbuffer<[u32]>,
	propose_likelihoods: Subbuffer<[f64]>,

	/// Unlike in the CPU likelihood, this field is essential.  It tracks
	/// which nodes were updated in the on-GPU buffer.  As such, it acts as
	/// the `edited` field in `SkVec`.
	updated_nodes_cache: Vec<usize>,
}

mod propose {
	vulkano_shaders::shader! {
		ty: "compute",
		path: "src/likelihood/gpu/propose.glsl",
	}
}

mod reject {
	vulkano_shaders::shader! {
		ty: "compute",
		path: "src/likelihood/gpu/reject.glsl",
	}
}

mod stage {
	vulkano_shaders::shader! {
		ty: "compute",
		path: "src/likelihood/gpu/stage.glsl",
	}
}

macro_rules! make_pipeline {
	($mod:ident, $device:ident, $layout:ident) => {{
		let entry_point = $mod::load($device.clone())?
			.entry_point("main")
			.ok_or(anyhow!("Entrypoint not fonud"))?;

		let stage =
			PipelineShaderStageCreateInfo::new(entry_point.clone());

		ComputePipeline::new(
			$device.clone(),
			None,
			ComputePipelineCreateInfo::stage_layout(
				stage,
				$layout.clone(),
			),
		)
	}};
}

impl LikelihoodTrait<4> for GpuLikelihood {
	fn propose(
		&mut self,
		nodes: &[usize],
		children: &[usize],
		transitions: &[Transition<4>],
	) -> Result<f64> {
		self.updated_nodes_cache = nodes.to_vec();

		let mut update_nodes = self.update_nodes.write()?;
		for (i, node) in nodes.iter().enumerate() {
			update_nodes[i] = *node as u32;
		}
		drop(update_nodes);

		let mut propose_nodes_length =
			self.update_nodes_length.write()?;
		*propose_nodes_length = nodes.len() as u32;
		drop(propose_nodes_length);

		let mut propose_children = self.propose_children.write()?;
		for (i, child) in children.iter().enumerate() {
			propose_children[i] = *child as u32;
		}
		drop(propose_children);

		let mut propose_transitions =
			self.propose_transitions.write()?;
		for (i, transition) in transitions.iter().enumerate() {
			propose_transitions[i] = *transition;
		}
		drop(propose_transitions);

		let future = sync::now(self.device.clone())
			.then_execute(
				self.queue.clone(),
				self.propose_command.clone(),
			)?
			.then_signal_fence_and_flush()?;

		future.wait(None)?;

		let output = self.propose_likelihoods.read()?;
		Ok(output.iter().map(|v| v.ln()).sum())
	}

	fn accept(&mut self) -> Result<()> {
		// `propose` changes the state to how it should be after the
		// update, so this is all what's needed to accept.
		self.updated_nodes_cache.clear();

		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		// This happens when an operator rejects prematurely without
		// making a suggestion.
		if self.updated_nodes_cache.is_empty() {
			return Ok(());
		}

		let future = sync::now(self.device.clone())
			.then_execute(
				self.queue.clone(),
				self.reject_command.clone(),
			)?
			.then_signal_fence_and_flush()?;

		future.wait(None)?;

		self.updated_nodes_cache.clear();

		Ok(())
	}
}

impl GpuLikelihood {
	pub fn new(sites: Vec<Vec<Row<4>>>) -> Result<Self> {
		let num_sites = sites.len();
		let num_leaves = sites[0].len();

		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;
		// A SkVec-like structure
		let mut probabilities = vec![];
		// The mask for probabilities.  32-bit integer in the smallest
		// int type on the GPU.
		let mut masks: Vec<u32> = vec![];
		for column in sites {
			for row in column {
				masks.push(0);
				probabilities.push(row);
				probabilities.push(Row::default());
			}
			for _ in 0..num_internals {
				masks.push(0);
				probabilities.push(Row::default());
				probabilities.push(Row::default());
			}
		}

		let library = VulkanLibrary::new()?;

		let instance =
			Instance::new(library, InstanceCreateInfo::default())?;

		let physical_device = instance
			.enumerate_physical_devices()?
			.next()
			.ok_or(anyhow!("No devices found"))?;

		let queue_family_index = physical_device
			.queue_family_properties()
			.iter()
			.enumerate()
			.position(|(_, queue_family_properties)| {
				queue_family_properties
					.queue_flags
					.contains(QueueFlags::COMPUTE)
			})
			.unwrap() as u32;

		let (device, mut queues) = Device::new(
			physical_device,
			DeviceCreateInfo {
				queue_create_infos: vec![QueueCreateInfo {
					queue_family_index,
					..Default::default()
				}],
				enabled_features: DeviceFeatures {
					shader_float64: true,
					..Default::default()
				},
				..Default::default()
			},
		)?;

		let queue = queues.next().unwrap();

		let memory_allocator = Arc::new(
			StandardMemoryAllocator::new_default(device.clone()),
		);

		let ds_allocator =
			Arc::new(StandardDescriptorSetAllocator::new(
				device.clone(),
				Default::default(),
			));

		let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
			device.clone(),
			StandardCommandBufferAllocatorCreateInfo::default(),
		));

		let push_constant = propose::PushConstants {
			num_nodes: num_nodes as u32,
			num_sites: num_sites as u32,
		};

		let push_constant_range = PushConstantRange {
			stages: ShaderStages::COMPUTE,
			offset: 0,
			size: size_of_val(&push_constant) as u32,
		};

		let layout_info = PipelineDescriptorSetLayoutCreateInfo {
			flags: PipelineLayoutCreateFlags::empty(),
			set_layouts: vec![
				dsl_probabilities(),
				dsl_update_nodes(),
				dsl_propose(),
				dsl_stage(),
			],
			push_constant_ranges: vec![push_constant_range],
		}
		.into_pipeline_layout_create_info(device.clone())?;
		let layout = PipelineLayout::new(device.clone(), layout_info)?;

		let propose_pipeline = make_pipeline!(propose, device, layout)?;
		let reject_pipeline = make_pipeline!(reject, device, layout)?;
		let stage_pipeline = make_pipeline!(stage, device, layout)?;

		let probabilities_buffer: Subbuffer<[Row<4>]> =
			Buffer::new_slice(
				memory_allocator.clone(),
				BufferCreateInfo {
					usage: BufferUsage::STORAGE_BUFFER,
					..Default::default()
				},
				AllocationCreateInfo {
					memory_type_filter:
						MemoryTypeFilter::PREFER_DEVICE,
					..Default::default()
				},
				probabilities.len() as u64,
			)?;
		let probabilities_masks: Subbuffer<[u32]> = Buffer::new_slice(
			memory_allocator.clone(),
			BufferCreateInfo {
				usage: BufferUsage::STORAGE_BUFFER,
				..Default::default()
			},
			AllocationCreateInfo {
				memory_type_filter:
					MemoryTypeFilter::PREFER_DEVICE,
				..Default::default()
			},
			masks.len() as u64,
		)?;

		let dsl_probabilities = DescriptorSetLayout::new(
			device.clone(),
			dsl_probabilities(),
		)?;
		let ds_probabilities = DescriptorSet::new(
			ds_allocator.clone(),
			dsl_probabilities.clone(),
			[
				WriteDescriptorSet::buffer(
					0,
					probabilities_buffer,
				),
				WriteDescriptorSet::buffer(
					1,
					probabilities_masks,
				),
			],
			[],
		)?;

		// Reject command
		let update_nodes: Subbuffer<[u32]> = Buffer::new_slice(
			memory_allocator.clone(),
			BufferCreateInfo {
				usage: BufferUsage::STORAGE_BUFFER,
				..Default::default()
			},
			AllocationCreateInfo {
				memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
					| MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
				..Default::default()
			},
			num_internals as u64,
		)?;
		let update_nodes_length: Subbuffer<u32> = Buffer::new_sized(
			memory_allocator.clone(),
			BufferCreateInfo {
				usage: BufferUsage::UNIFORM_BUFFER,
				..Default::default()
			},
			AllocationCreateInfo {
				memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
					| MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
				..Default::default()
			},
		)?;

		let dsl_update_nodes = DescriptorSetLayout::new(
			device.clone(),
			dsl_update_nodes(),
		)?;
		let ds_update_nodes = DescriptorSet::new(
			ds_allocator.clone(),
			dsl_update_nodes.clone(),
			[
				WriteDescriptorSet::buffer(
					0,
					update_nodes.clone(),
				),
				WriteDescriptorSet::buffer(
					1,
					update_nodes_length.clone(),
				),
			],
			[],
		)?;

		let mut command_buffer_builder =
			AutoCommandBufferBuilder::primary(
				command_buffer_allocator.clone(),
				queue.queue_family_index(),
				CommandBufferUsage::MultipleSubmit,
			)?;

		let num_groups = (num_sites + 63) / 64;
		let work_group_counts = [num_groups as u32, 1, 1];

		let reject_cmd_buffer = command_buffer_builder
			.bind_pipeline_compute(reject_pipeline.clone())?
			.push_constants(
				propose_pipeline.layout().clone(),
				0,
				push_constant,
			)?
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				reject_pipeline.layout().clone(),
				0u32,
				ds_probabilities.clone(),
			)?
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				reject_pipeline.layout().clone(),
				1u32,
				ds_update_nodes.clone(),
			)?;

		// TODO: safety
		unsafe { reject_cmd_buffer.dispatch(work_group_counts)? };

		let reject_command = command_buffer_builder.build()?;

		let propose_transitions: Subbuffer<[Transition<4>]> = Buffer::new_slice(
			memory_allocator.clone(),
			BufferCreateInfo {
				usage: BufferUsage::STORAGE_BUFFER,
				..Default::default()
			},
			AllocationCreateInfo {
				memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
					| MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
				..Default::default()
			},
			(num_internals * 2) as u64,
		)?;
		let propose_children: Subbuffer<[u32]> = Buffer::new_slice(
			memory_allocator.clone(),
			BufferCreateInfo {
				usage: BufferUsage::STORAGE_BUFFER,
				..Default::default()
			},
			AllocationCreateInfo {
				memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
					| MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
				..Default::default()
			},
			(num_internals * 2) as u64,
		)?;
		let propose_likelihoods: Subbuffer<[f64]> = Buffer::new_slice(
			memory_allocator.clone(),
			BufferCreateInfo {
				usage: BufferUsage::STORAGE_BUFFER,
				..Default::default()
			},
			AllocationCreateInfo {
				memory_type_filter:
					MemoryTypeFilter::HOST_RANDOM_ACCESS,
				..Default::default()
			},
			num_sites as u64,
		)?;

		let dsl_propose = DescriptorSetLayout::new(
			device.clone(),
			dsl_propose(),
		)?;
		let ds_propose = DescriptorSet::new(
			ds_allocator.clone(),
			dsl_propose.clone(),
			[
				WriteDescriptorSet::buffer(
					0,
					propose_transitions.clone(),
				),
				WriteDescriptorSet::buffer(
					1,
					propose_children.clone(),
				),
				WriteDescriptorSet::buffer(
					2,
					propose_likelihoods.clone(),
				),
			],
			[],
		)?;

		let mut command_buffer_builder =
			AutoCommandBufferBuilder::primary(
				command_buffer_allocator.clone(),
				queue.queue_family_index(),
				CommandBufferUsage::MultipleSubmit,
			)?;

		let cmd = command_buffer_builder
			.bind_pipeline_compute(propose_pipeline.clone())?
			.push_constants(
				propose_pipeline.layout().clone(),
				0,
				push_constant,
			)?
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				propose_pipeline.layout().clone(),
				0u32,
				ds_probabilities.clone(),
			)?
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				propose_pipeline.layout().clone(),
				1u32,
				ds_update_nodes.clone(),
			)?
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				propose_pipeline.layout().clone(),
				2u32,
				ds_propose,
			)?;

		// TODO: safety
		let cmd = unsafe { cmd.dispatch(work_group_counts) };
		cmd?;

		let propose_command = command_buffer_builder.build()?;

		let stage_probabilities: Subbuffer<[Row<4>]> =
			Buffer::from_iter(
				memory_allocator.clone(),
				BufferCreateInfo {
					usage: BufferUsage::STORAGE_BUFFER,
					..Default::default()
				},
				AllocationCreateInfo {
					memory_type_filter:
						MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
					..Default::default()
				},
				probabilities,
			)?;
		let stage_masks = Buffer::from_iter(
			memory_allocator.clone(),
			BufferCreateInfo {
				usage: BufferUsage::STORAGE_BUFFER,
				..Default::default()
			},
			AllocationCreateInfo {
				memory_type_filter:
					MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
				..Default::default()
			},
			masks,
		)?;

		let dsl_stage =
			DescriptorSetLayout::new(device.clone(), dsl_stage())?;
		let ds_stage = DescriptorSet::new(
			ds_allocator.clone(),
			dsl_stage.clone(),
			[
				WriteDescriptorSet::buffer(
					0,
					stage_probabilities,
				),
				WriteDescriptorSet::buffer(
					1,
					stage_masks.clone(),
				),
			],
			[],
		)?;

		let mut command_buffer_builder =
			AutoCommandBufferBuilder::primary(
				command_buffer_allocator.clone(),
				queue.queue_family_index(),
				CommandBufferUsage::MultipleSubmit,
			)?;

		let cmd = command_buffer_builder
			.bind_pipeline_compute(stage_pipeline.clone())?
			.push_constants(
				propose_pipeline.layout().clone(),
				0,
				push_constant,
			)?
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				stage_pipeline.layout().clone(),
				0u32,
				ds_probabilities.clone(),
			)?
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				stage_pipeline.layout().clone(),
				3u32,
				ds_stage,
			)?;

		// TODO: safety
		let cmd = unsafe { cmd.dispatch(work_group_counts) };
		cmd?;

		let stage_command = command_buffer_builder.build()?;

		let future = sync::now(device.clone())
			.then_execute(queue.clone(), stage_command.clone())?
			.then_signal_fence_and_flush()?;

		future.wait(None)?;

		Ok(GpuLikelihood {
			device,
			queue,

			reject_command,
			update_nodes,
			update_nodes_length,

			propose_command,
			propose_transitions,
			propose_children,
			propose_likelihoods,

			updated_nodes_cache: Vec::new(),
		})
	}
}

fn dsl_binding(dtype: DescriptorType) -> DescriptorSetLayoutBinding {
	DescriptorSetLayoutBinding {
		stages: ShaderStages::COMPUTE,
		..DescriptorSetLayoutBinding::descriptor_type(dtype)
	}
}

fn dsl_probabilities() -> DescriptorSetLayoutCreateInfo {
	DescriptorSetLayoutCreateInfo {
		bindings: BTreeMap::from([
			(0, dsl_binding(DescriptorType::StorageBuffer)),
			(1, dsl_binding(DescriptorType::StorageBuffer)),
		]),
		..Default::default()
	}
}

fn dsl_update_nodes() -> DescriptorSetLayoutCreateInfo {
	DescriptorSetLayoutCreateInfo {
		bindings: BTreeMap::from([
			(0, dsl_binding(DescriptorType::StorageBuffer)),
			(1, dsl_binding(DescriptorType::UniformBuffer)),
		]),
		..Default::default()
	}
}

fn dsl_stage() -> DescriptorSetLayoutCreateInfo {
	DescriptorSetLayoutCreateInfo {
		bindings: BTreeMap::from([
			(0, dsl_binding(DescriptorType::StorageBuffer)),
			(1, dsl_binding(DescriptorType::StorageBuffer)),
		]),
		..Default::default()
	}
}

fn dsl_propose() -> DescriptorSetLayoutCreateInfo {
	DescriptorSetLayoutCreateInfo {
		bindings: BTreeMap::from([
			(0, dsl_binding(DescriptorType::StorageBuffer)),
			(1, dsl_binding(DescriptorType::StorageBuffer)),
			(2, dsl_binding(DescriptorType::StorageBuffer)),
		]),
		..Default::default()
	}
}
