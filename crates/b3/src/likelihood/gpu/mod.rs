use anyhow::{anyhow, Result};
use vulkano::{
	buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
	command_buffer::{
		allocator::{
			CommandBufferAllocator, StandardCommandBufferAllocator,
			StandardCommandBufferAllocatorCreateInfo,
		},
		AutoCommandBufferBuilder, CommandBufferUsage,
		PrimaryAutoCommandBuffer,
	},
	descriptor_set::{
		allocator::DescriptorSetAllocator,
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
	memory_allocator: Arc<StandardMemoryAllocator>,
	descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
	command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
	propose_pipeline: Arc<ComputePipeline>,
	descriptor_set_0: Arc<DescriptorSet>,

	reject_cmd: Arc<PrimaryAutoCommandBuffer>,
	reject_nodes_buffer: Subbuffer<[u32]>,
	reject_nodes_len_buffer: Subbuffer<u32>,

	/// Unlike in the CPU likelihood, this field is essential.  It tracks
	/// which nodes were updated in the on-GPU buffer.  As such, it acts as
	/// the `edited` field in `SkVec`.
	updated_nodes: Vec<usize>,

	/// The length of each sequence
	num_sites: usize,
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

impl LikelihoodTrait<4> for GpuLikelihood {
	fn propose(
		&mut self,
		nodes: &[usize],
		transitions: &[Transition<4>],
		children: &[usize],
	) -> f64 {
		self.updated_nodes = nodes.to_vec();

		let pipeline_layout = self.propose_pipeline.layout();
		let descriptor_set_layouts = pipeline_layout.set_layouts();

		let nodes_buffer = Buffer::from_iter(
			self.memory_allocator.clone(),
			BufferCreateInfo {
				usage: BufferUsage::STORAGE_BUFFER,
				..Default::default()
			},
			AllocationCreateInfo {
				memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
					| MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
				..Default::default()
			},
			nodes.iter().map(|v| *v as u32),
		).unwrap();
		let substitutions_buffer = Buffer::from_iter(
			self.memory_allocator.clone(),
			BufferCreateInfo {
				usage: BufferUsage::STORAGE_BUFFER,
				..Default::default()
			},
			AllocationCreateInfo {
				memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
					| MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
				..Default::default()
			},
			transitions.iter().copied(),
		).unwrap();
		let children_buffer = Buffer::from_iter(
			self.memory_allocator.clone(),
			BufferCreateInfo {
				usage: BufferUsage::STORAGE_BUFFER,
				..Default::default()
			},
			AllocationCreateInfo {
				memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
					| MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
				..Default::default()
			},
			children.iter().map(|v| *v as u32),
		).unwrap();
		let likelihoods_buffer = Buffer::from_iter(
			self.memory_allocator.clone(),
			BufferCreateInfo {
				usage: BufferUsage::STORAGE_BUFFER,
				..Default::default()
			},
			AllocationCreateInfo {
				memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
					| MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
				..Default::default()
			},
			(0..self.num_sites).map(|_| 0.0f64),
		).unwrap();

		let descriptor_set_layout_1 =
			descriptor_set_layouts.get(1).unwrap();
		let descriptor_set_1 = DescriptorSet::new(
			self.descriptor_set_allocator.clone(),
			descriptor_set_layout_1.clone(),
			[
				WriteDescriptorSet::buffer(0, nodes_buffer),
				WriteDescriptorSet::buffer(
					1,
					substitutions_buffer,
				),
				WriteDescriptorSet::buffer(2, children_buffer),
				WriteDescriptorSet::buffer(
					3,
					likelihoods_buffer.clone(),
				),
			],
			[],
		)
		.unwrap();

		let mut command_buffer_builder =
			AutoCommandBufferBuilder::primary(
				self.command_buffer_allocator.clone(),
				self.queue.queue_family_index(),
				CommandBufferUsage::OneTimeSubmit,
			)
			.unwrap();

		let num_groups = (self.num_sites + 63) / 64;
		let work_group_counts = [num_groups as u32, 1, 1];

		let cmd = command_buffer_builder
			.bind_pipeline_compute(self.propose_pipeline.clone())
			.unwrap()
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				self.propose_pipeline.layout().clone(),
				0u32,
				self.descriptor_set_0.clone(),
			)
			.unwrap()
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				self.propose_pipeline.layout().clone(),
				1u32,
				descriptor_set_1,
			)
			.unwrap();

		// TODO: safety
		let cmd = unsafe { cmd.dispatch(work_group_counts) };
		cmd.unwrap();

		let command_buffer = command_buffer_builder.build().unwrap();

		let future = sync::now(self.device.clone())
			.then_execute(self.queue.clone(), command_buffer)
			.unwrap()
			.then_signal_fence_and_flush()
			.unwrap();

		future.wait(None).unwrap();

		let output = likelihoods_buffer.read().unwrap();
		output.iter().map(|v| v.ln()).sum()
	}

	fn accept(&mut self) {
		// `propose` changes the state to how it should be after the
		// update, so this is all what's needed to accept.
		self.updated_nodes.clear();
	}

	fn reject(&mut self) {
		// This happens when an operator rejects prematurely without
		// making a suggestion.
		if self.updated_nodes.is_empty() {
			return;
		}

		let mut nodes = self.reject_nodes_buffer.write().unwrap();
		for (i, node) in self.updated_nodes.iter().enumerate() {
			nodes[i] = (*node) as u32;
		}
		drop(nodes);

		let mut length = self.reject_nodes_len_buffer.write().unwrap();
		*length = self.updated_nodes.len() as u32;
		drop(length);

		let future = sync::now(self.device.clone())
			.then_execute(
				self.queue.clone(),
				self.reject_cmd.clone(),
			)
			.unwrap()
			.then_signal_fence_and_flush()
			.unwrap();

		future.wait(None).unwrap();

		self.updated_nodes.clear();
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

		let num_rows_buffer = Buffer::from_data(
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
			(num_leaves * 2 - 1) as u32,
		)?;

		let probabilities_buffer = Buffer::from_iter(
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
			probabilities.clone(),
		)?;

		let masks_buffer = Buffer::from_iter(
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
			masks.clone(),
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

		let pipeline_layout = propose_pipeline.layout();
		let descriptor_set_layouts = pipeline_layout.set_layouts();
		let descriptor_set_layout_0 = descriptor_set_layouts[0].clone();
		let descriptor_set_0 = DescriptorSet::new(
			descriptor_set_allocator.clone(),
			descriptor_set_layout_0.clone(),
			[
				WriteDescriptorSet::buffer(0, num_rows_buffer),
				WriteDescriptorSet::buffer(
					1,
					probabilities_buffer,
				),
				WriteDescriptorSet::buffer(2, masks_buffer),
			],
			[],
		)?;

		// Reject command
		let reject_nodes_buffer: Subbuffer<[u32]> = Buffer::new_unsized(
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
		let reject_nodes_len_buffer: Subbuffer<u32> = Buffer::new_sized(
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

		let pipeline_layout = reject_pipeline.layout();
		let descriptor_set_layouts = pipeline_layout.set_layouts();
		let descriptor_set_layout_1 =
			descriptor_set_layouts.get(1).unwrap();
		let descriptor_set_1 = DescriptorSet::new(
			descriptor_set_allocator.clone(),
			descriptor_set_layout_1.clone(),
			[
				WriteDescriptorSet::buffer(
					0,
					reject_nodes_len_buffer.clone(),
				),
				WriteDescriptorSet::buffer(
					1,
					reject_nodes_len_buffer.clone(),
				),
			],
			[],
		)?;

		let mut command_buffer_builder =
			AutoCommandBufferBuilder::primary(
				command_buffer_allocator.clone(),
				queue.queue_family_index(),
				CommandBufferUsage::MultipleSubmit,
			)
			.unwrap();

		let num_groups = (num_sites + 63) / 64;
		let work_group_counts = [num_groups as u32, 1, 1];

		let reject_cmd_buffer = command_buffer_builder
			.bind_pipeline_compute(reject_pipeline.clone())?
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				reject_pipeline.layout().clone(),
				0u32,
				descriptor_set_0.clone(),
			)?
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				reject_pipeline.layout().clone(),
				1u32,
				descriptor_set_1,
			)?;
		unsafe { reject_cmd_buffer.dispatch(work_group_counts)? };

		let reject_cmd = command_buffer_builder.build()?;

		Ok(GpuLikelihood {
			device,
			queue,
			memory_allocator,
			descriptor_set_allocator,
			command_buffer_allocator,
			propose_pipeline,

			descriptor_set_0,

			reject_cmd,
			reject_nodes_buffer,
			reject_nodes_len_buffer,

			updated_nodes: Vec::new(),

			num_sites,
		})
	}
}
