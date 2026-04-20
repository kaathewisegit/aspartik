#include <cooperative_groups.h>

using namespace cooperative_groups;

__device__ f64 sum(const f64x4 v) {
    return v.x + v.y + v.z + v.w;
}

__device__ f64 dot(const f64x4 a, const f64x4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ f64x4 hadamard(const f64x4 a, const f64x4 b) {
	return make_f64x4(
		a.x * b.x,
		a.y * b.y,
		a.z * b.z,
		a.w * b.w
	);
}

#define BLOCK_SIZE 16 * 4

#define idx(edge) \
	((edge) * NUM_PATTERNS + pattern)

// Selector index
#define sidx(edge) \
	((edge) * 2 + (selectors[edge] & 1)) * NUM_PATTERNS + pattern

// Selector index, other half
#define soidx(edge) \
	((edge) * 2 + (selectors[edge] & 1 ^ 1)) * NUM_PATTERNS + pattern

// Selector for propagations
#define spidx(edge) \
	(sidx(edge)) * 4 + sub

// Gets the pattern index from the thread and block id
#define PATTERN_PRELUDE \
	u32 pattern = blockIdx.x * blockDim.x + threadIdx.x; \
	if (pattern >= NUM_PATTERNS) { \
		return; \
	} \

// # Variables
// - i: index of the update
// - sub: index of the pattern allele
#define CALCULATE_LEAF_PROJECTION \
	u8 leaf = leaves[idx(nodes[i])]; \
	f64 projection = 0.0; \
	f64x4 tv = transitions[i * 4 + sub]; \
	if (leaf == 0b0001) projection = tv.x; \
	else if (leaf == 0b0010) projection = tv.y; \
	else if (leaf == 0b0100) projection = tv.z; \
	else if (leaf == 0b1000) projection = tv.w; \
	else projection = 1.0; \
	projections[spidx(nodes[i])] = projection; \

// Update partial likelihoods for edges which go to leaves
entrypoint __launch_bounds__(BLOCK_SIZE)
void update_leaves(
	const u8* __restrict__ leaves,
	f64* __restrict__ projections,
	const u8* __restrict__ selectors,

	const u32* __restrict__ nodes,
	const f64x4* __restrict__ transitions
) {
	u32 pattern = blockIdx.x * blockDim.x + threadIdx.x;
	if (pattern >= NUM_PATTERNS) {
		return;
	}
	u32 sub = threadIdx.y;
	u32 i = blockIdx.y;

	CALCULATE_LEAF_PROJECTION
}

entrypoint __launch_bounds__(BLOCK_SIZE)
void propose(
	const u8* __restrict__ leaves,
	f64* __restrict__ projections,
	u8* __restrict__ scales,
	u32* __restrict__ scale_sums,
	const u8* __restrict__ selectors,

	const u32 num_updated_nodes,
	const u32* __restrict__ nodes,
	const u32* __restrict__ children,
	const f64x4* __restrict__ transitions,

	const u32 leaves_end,
	const u32 internals_start
) {
	u32 pattern = (blockIdx.x * blockDim.x + threadIdx.x) / 4;
	if (pattern >= NUM_PATTERNS) {
		return;
	}
	auto g = tiled_partition<4>(this_thread_block());
	u32 sub = g.thread_rank();
	u32 tile = threadIdx.x / 4;

	__shared__ f64 s_likelihood[BLOCK_SIZE];

	for (u32 i = 0; i < leaves_end; i++) {
		CALCULATE_LEAF_PROJECTION
	}

	u32 scale_sum = scale_sums[pattern];

	for (u32 i = internals_start; i < num_updated_nodes; i++) {
		u32 left_edge = children[(i - internals_start) * 2];
		u32 right_edge = children[(i - internals_start) * 2 + 1];
		u32 this_edge = nodes[i];
		u32 old_scale = scales[soidx(this_edge)];

		// thread-local likelihood
		f64 l_likelihood = projections[spidx(left_edge)] *
			projections[spidx(right_edge)];
		s_likelihood[tile * 4 + sub] = l_likelihood;

		g.sync();

		u32 should_scale = s_likelihood[tile * 4 + 0] < SCALE_THRESHOLD
			&& s_likelihood[tile * 4 + 1] < SCALE_THRESHOLD
			&& s_likelihood[tile * 4 + 2] < SCALE_THRESHOLD
			&& s_likelihood[tile * 4 + 3] < SCALE_THRESHOLD;

		if (should_scale) {
			s_likelihood[tile * 4 + sub] *= SCALE_MULT;

			g.sync();
		}

		if (sub == 0) {
			scales[sidx(this_edge)] = should_scale;
			if (old_scale == 0 && should_scale == 1) {
				scale_sum += SCALE_LN;
			} else if (old_scale == 1 && should_scale == 0) {
				scale_sum -= SCALE_LN;
			}
		}

		// rebuild the likelihood from the 4 neighboring threads
		auto likelihood = make_f64x4(
			s_likelihood[tile * 4 + 0],
			s_likelihood[tile * 4 + 1],
			s_likelihood[tile * 4 + 2],
			s_likelihood[tile * 4 + 3]
		);

		f64 projection = dot(
			transitions[i * 4 + sub],
			likelihood
		);

		projections[spidx(this_edge)] = projection;
	}

	if (sub == 0) {
		scale_sums[pattern] = scale_sum;
	}
}

entrypoint __launch_bounds__(32)
void update_likelihoods(
	const f64x4* __restrict__ projections,
	f64* __restrict__ likelihoods,
	u8* __restrict__ scales,
	u32* scale_sums,
	const u8* __restrict__ selectors,

	u32 root,
	u32 left_child,
	u32 right_child,
	f64x4 frequencies
) {
	PATTERN_PRELUDE

	f64x4 pre_likelihood = hadamard(
		projections[sidx(left_child)],
		projections[sidx(right_child)]
	);
	f64x4 likelihood = hadamard(
		pre_likelihood,
		frequencies
	);

	f64 sum = likelihood.x + likelihood.y + likelihood.z + likelihood.w;
	likelihoods[pattern] = log(sum);

	scales[sidx(root)] = 0;
	if (scales[soidx(root)]) {
		scale_sums[pattern] -= SCALE_LN;
	}
}
