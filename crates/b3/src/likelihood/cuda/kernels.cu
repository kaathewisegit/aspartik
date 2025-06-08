#include "typedefs.h"

// These are used in __launch_bounds__, which define maximum number of threads
// per block.  The less threads there are, the more registers each kernel can
// use.
#define BLOCK_SIZE_SEQ 32
#define BLOCK_SIZE_PAR 128

typedef struct {
	f64x4 a, c, g, t;
} Transition;

__device__ double dot(const f64x4 a, const f64x4 b) {
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

__device__ f64x4 apply(
	const Transition transition,
	const f64x4 vector
) {
	return make_f64x4(
		dot(transition.a, vector),
		dot(transition.c, vector),
		dot(transition.g, vector),
		dot(transition.t, vector)
	);
}

#define idx(edge) \
	((edge) * num_sites + site)

// Gets the site index from the thread and block id
#define SITE_PRELUDE \
	u32 site = blockIdx.x * blockDim.x + threadIdx.x; \
	if (site >= num_sites) { \
		return; \
	} \

#define PAR_BLOCK_PRELUDE \
	SITE_PRELUDE \
	u32 i = blockIdx.y; \

extern "C" __global__ __launch_bounds__(BLOCK_SIZE_PAR)
void update_leaves(
	const u32 num_sites,

	const u32* __restrict__ edges,
	const u32* __restrict__ nodes,
	const Transition* __restrict__ transitions,

	const f64x4* __restrict__ leaves,
	f64x4* __restrict__ projections
) {
	PAR_BLOCK_PRELUDE

	f64x4 projection = apply(transitions[i], leaves[idx(nodes[i])]);
	projections[idx(edges[i])] = projection;
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE_PAR)
void update_internals(
	const u32 num_sites,
	const u32 num_leaves,

	const f64x4* __restrict__ leaves,
	f64x4* __restrict__ projections,

	const u32* __restrict__ edges,
	const u32* __restrict__ nodes,
	const Transition* __restrict__ transitions
) {
	PAR_BLOCK_PRELUDE

	u32 left_edge = (nodes[i] - num_leaves) * 2;
	u32 right_edge = left_edge + 1;

	f64x4 likelihood = hadamard(
		projections[idx(left_edge)],
		projections[idx(right_edge)]
	);

	f64x4 projection = apply(
		transitions[i],
		likelihood
	);

	projections[idx(edges[i])] = projection;
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE_SEQ)
void propose(
	const u32 num_sites,
	const u32 num_leaves,

	const f64x4* __restrict__ leaves,
	f64x4* __restrict__ projections,
	double* __restrict__ likelihoods,

	const u32 num_updated_nodes,
	const u32* __restrict__ nodes,
	const u32* __restrict__ edges,
	const Transition* __restrict__ transitions,

	const u32 leaves_end,
	const u32 internals_start,
	const u32 root
) {
	SITE_PRELUDE

	for (u32 i = 0; i < leaves_end; i++) {
		f64x4 projection = apply(
			transitions[i],
			leaves[idx(nodes[i])]
		);

		projections[idx(edges[i])] = projection;
	}

	for (u32 i = internals_start; i < num_updated_nodes; i++) {
		u32 left_edge = (nodes[i] - num_leaves) * 2;
		u32 right_edge = left_edge + 1;

		f64x4 likelihood = hadamard(
			projections[idx(left_edge)],
			projections[idx(right_edge)]
		);

		f64x4 projection = apply(
			transitions[i],
			likelihood
		);

		projections[idx(edges[i])] = projection;
	}

	u32 left_root_edge = (root - num_leaves) * 2;
	u32 right_root_edge = left_root_edge + 1;

	f64x4 likelihood = hadamard(
		projections[idx(left_root_edge)],
		projections[idx(right_root_edge)]
	);

	double sum = likelihood.x + likelihood.y + likelihood.z + likelihood.w;
	likelihoods[site] = log(sum);
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE_PAR)
void accept(
	const u32 num_sites,

	const f64x4* __restrict__ projections,
	f64x4* __restrict__ projections_backup,

	const u32 num_updated_nodes,
	const u32* __restrict__ edges
) {
	PAR_BLOCK_PRELUDE

	u32 proj_idx = idx(edges[i]);
	projections_backup[proj_idx] = projections[proj_idx];
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE_PAR)
void reject(
	const u32 num_sites,

	f64x4* __restrict__ projections,
	const f64x4* __restrict__ projections_backup,

	const u32 num_updated_nodes,
	const u32* __restrict__ edges
) {
	PAR_BLOCK_PRELUDE

	u32 proj_idx = idx(edges[i]);
	projections[proj_idx] = projections_backup[proj_idx];
}
