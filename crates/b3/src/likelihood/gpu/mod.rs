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
		allocator::StandardDescriptorSetAllocator, DescriptorSet,
		WriteDescriptorSet,
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
		layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline,
		Pipeline, PipelineBindPoint, PipelineLayout,
		PipelineShaderStageCreateInfo,
	},
	sync::{self, GpuFuture},
	VulkanLibrary,
};

use std::sync::Arc;

use super::{LikelihoodTrait, Row, Transition};

pub struct GpuLikelihood {
	// TODO: bench and see if allocators and such should be preserved here
	// buffers:
	// - one array for final likelihoods
	// - number of nodes
	device: Arc<Device>,
	queue: Arc<Queue>,

	reject_command: Arc<PrimaryAutoCommandBuffer>,
	reject_nodes: Subbuffer<[u32]>,
	reject_nodes_length: Subbuffer<u32>,

	propose_command: Arc<PrimaryAutoCommandBuffer>,
	propose_nodes: Subbuffer<[u32]>,
	propose_transitions: Subbuffer<[Transition<4>]>,
	propose_children: Subbuffer<[u32]>,
	propose_likelihoods: Subbuffer<[f64]>,
	propose_nodes_length: Subbuffer<u32>,

	/// Unlike in the CPU likelihood, this field is essential.  It tracks
	/// which nodes were updated in the on-GPU buffer.  As such, it acts as
	/// the `edited` field in `SkVec`.
	updated_nodes: Vec<usize>,
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

impl LikelihoodTrait<4> for GpuLikelihood {
	fn propose(
		&mut self,
		nodes: &[usize],
		transitions: &[Transition<4>],
		children: &[usize],
	) -> Result<f64> {
		self.updated_nodes = nodes.to_vec();

		let mut accept_nodes = self.propose_nodes.write()?;
		for (i, node) in nodes.iter().enumerate() {
			accept_nodes[i] = *node as u32;
		}
		drop(accept_nodes);

		let mut accept_nodes_length =
			self.propose_nodes_length.write()?;
		*accept_nodes_length = nodes.len() as u32;
		drop(accept_nodes_length);

		let mut accept_children = self.propose_children.write()?;
		for (i, child) in children.iter().enumerate() {
			accept_children[i] = *child as u32;
		}
		drop(accept_children);

		let mut accept_transitions =
			self.propose_transitions.write()?;
		for (i, transition) in transitions.iter().enumerate() {
			accept_transitions[i] = *transition;
		}
		drop(accept_transitions);

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
		self.updated_nodes.clear();

		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		// This happens when an operator rejects prematurely without
		// making a suggestion.
		if self.updated_nodes.is_empty() {
			return Ok(());
		}

		let mut nodes = self.reject_nodes.write()?;
		for (i, node) in self.updated_nodes.iter().enumerate() {
			nodes[i] = (*node) as u32;
		}
		drop(nodes);

		let mut length = self.reject_nodes_length.write()?;
		*length = self.updated_nodes.len() as u32;
		drop(length);

		let future = sync::now(self.device.clone())
			.then_execute(
				self.queue.clone(),
				self.reject_command.clone(),
			)?
			.then_signal_fence_and_flush()?;

		future.wait(None)?;

		self.updated_nodes.clear();

		Ok(())
	}
}

impl GpuLikelihood {
	pub fn new(sites: Vec<Vec<Row<4>>>) -> Result<Self> {
		let num_sites = sites.len();
		let num_leaves = sites[0].len();

		let num_internals = num_leaves - 1;
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

		let common_num_rows: Subbuffer<u32> = Buffer::new_sized(
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
		)?;
		let common_probabilities: Subbuffer<[Row<4>]> =
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
		let common_masks: Subbuffer<[u32]> = Buffer::new_slice(
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

		let descriptor_set_allocator =
			Arc::new(StandardDescriptorSetAllocator::new(
				device.clone(),
				Default::default(),
			));

		let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
			device.clone(),
			StandardCommandBufferAllocatorCreateInfo::default(),
		));

		#[rustfmt::skip]
		macro_rules! make_pipeline {
			($mod:ident) => {{

		let shader = $mod::load(device.clone())?
			.entry_point("main")
			.ok_or(anyhow!("Entrypoint not fonud"))?;
		let stage = PipelineShaderStageCreateInfo::new(
			shader.clone(),
		);
		let layout = PipelineLayout::new(
			device.clone(),
			PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
			.into_pipeline_layout_create_info(device.clone())?,
		)?;

		ComputePipeline::new(
			device.clone(),
			None,
			ComputePipelineCreateInfo::stage_layout(stage, layout),
		)?

			}};
		}

		let reject_pipeline = make_pipeline!(reject);
		let propose_pipeline = make_pipeline!(propose);
		let stage_pipeline = make_pipeline!(stage);

		let descriptor_set_layout_common =
			propose_pipeline.layout().set_layouts()[0].clone();
		let descriptor_set_common = DescriptorSet::new(
			descriptor_set_allocator.clone(),
			descriptor_set_layout_common.clone(),
			[
				WriteDescriptorSet::buffer(0, common_num_rows),
				WriteDescriptorSet::buffer(
					1,
					common_probabilities,
				),
				WriteDescriptorSet::buffer(2, common_masks),
			],
			[],
		)?;

		// Reject command
		let reject_nodes: Subbuffer<[u32]> = Buffer::new_slice(
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
		let reject_nodes_length: Subbuffer<u32> = Buffer::new_sized(
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
		)?;

		let descriptor_set_layout_reject =
			reject_pipeline.layout().set_layouts()[1].clone();
		let descriptor_set_reject = DescriptorSet::new(
			descriptor_set_allocator.clone(),
			descriptor_set_layout_reject.clone(),
			[
				WriteDescriptorSet::buffer(
					0,
					reject_nodes.clone(),
				),
				WriteDescriptorSet::buffer(
					1,
					reject_nodes_length.clone(),
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
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				reject_pipeline.layout().clone(),
				0u32,
				descriptor_set_common.clone(),
			)?
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				reject_pipeline.layout().clone(),
				1u32,
				descriptor_set_reject,
			)?;
		unsafe { reject_cmd_buffer.dispatch(work_group_counts)? };

		let reject_command = command_buffer_builder.build()?;

		let propose_nodes: Subbuffer<[u32]> = Buffer::new_slice(
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
				memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
					| MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
				..Default::default()
			},
			num_sites as u64,
		)?;
		let propose_nodes_length: Subbuffer<u32> = Buffer::new_sized(
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
		)?;

		let descriptor_set_layout_propose =
			propose_pipeline.layout().set_layouts()[1].clone();
		let descriptor_set_propose = DescriptorSet::new(
			descriptor_set_allocator.clone(),
			descriptor_set_layout_propose.clone(),
			[
				WriteDescriptorSet::buffer(
					0,
					propose_nodes.clone(),
				),
				WriteDescriptorSet::buffer(
					1,
					propose_transitions.clone(),
				),
				WriteDescriptorSet::buffer(
					2,
					propose_children.clone(),
				),
				WriteDescriptorSet::buffer(
					3,
					propose_likelihoods.clone(),
				),
				WriteDescriptorSet::buffer(
					4,
					propose_nodes_length.clone(),
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
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				propose_pipeline.layout().clone(),
				0u32,
				descriptor_set_common.clone(),
			)?
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				propose_pipeline.layout().clone(),
				1u32,
				descriptor_set_propose,
			)?;

		// TODO: safety
		let cmd = unsafe { cmd.dispatch(work_group_counts) };
		cmd?;

		let propose_command = command_buffer_builder.build()?;

		let stage_num_rows = Buffer::from_data(
			memory_allocator.clone(),
			BufferCreateInfo {
				usage: BufferUsage::STORAGE_BUFFER,
				..Default::default()
			},
			AllocationCreateInfo {
				memory_type_filter:
					MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
				..Default::default()
			},
			(num_leaves + num_internals) as u32,
		)?;
		let stage_probabilities: Subbuffer<[Row<4>]> =
			Buffer::from_iter(
				memory_allocator.clone(),
				BufferCreateInfo {
					usage: BufferUsage::STORAGE_BUFFER,
					..Default::default()
				},
				AllocationCreateInfo {
					memory_type_filter:
						MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
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
					MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
				..Default::default()
			},
			masks,
		)?;

		let descriptor_set_layout_stage =
			stage_pipeline.layout().set_layouts()[1].clone();
		let descriptor_set_stage = DescriptorSet::new(
			descriptor_set_allocator.clone(),
			descriptor_set_layout_stage.clone(),
			[
				WriteDescriptorSet::buffer(0, stage_num_rows),
				WriteDescriptorSet::buffer(
					1,
					stage_probabilities,
				),
				WriteDescriptorSet::buffer(
					2,
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
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				stage_pipeline.layout().clone(),
				0u32,
				descriptor_set_common.clone(),
			)?
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				stage_pipeline.layout().clone(),
				1u32,
				descriptor_set_stage,
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
			reject_nodes,
			reject_nodes_length,

			propose_command,
			propose_nodes,
			propose_transitions,
			propose_children,
			propose_likelihoods,
			propose_nodes_length,

			updated_nodes: Vec::new(),
		})
	}
}
