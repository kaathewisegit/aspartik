#include "typedefs.h"

typedef struct {
	f64x4 a, c, g, t;
} Transition;

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

#define BLOCK_SIZE 16 * 4

#define idx(edge) \
	((edge) * num_sites + site)

#define sidx(edge) \
	((edge) * num_sites + site) * 4 + sub

// Gets the site index from the thread and block id
#define SITE_PRELUDE \
	u32 site = blockIdx.x * blockDim.x + threadIdx.x; \
	if (site >= num_sites) { \
		return; \
	} \

// # Variables
// - i: index of the update
// - sub: index of the site allele
#define CALCULATE_LEAF_PROJECTION \
	f64 projection = dot( \
		transitions[i * 4 + sub], \
		leaves[idx(nodes[i])] \
	); \
	projections[sidx(edges[i])] = projection; \

entrypoint __launch_bounds__(BLOCK_SIZE)
void update_leaves(
	const u32 num_sites,

	const f64x4* restrict leaves,
	f64* restrict projections,

	const u32* restrict nodes,
	const u32* restrict edges,
	const f64x4* restrict transitions
) {
	u32 site = blockIdx.x * blockDim.x + threadIdx.x;
	if (site >= num_sites) {
		return;
	}
	u32 sub = threadIdx.y;
	u32 i = blockIdx.y;

	CALCULATE_LEAF_PROJECTION
}

entrypoint __launch_bounds__(BLOCK_SIZE)
void propose(
	const u32 num_sites,
	const u32 num_leaves,

	const f64x4* restrict leaves,
	f64* restrict projections,

	const u32 num_updated_nodes,
	const u32* restrict nodes,
	const u32* restrict edges,
	const f64x4* restrict transitions,

	const u32 leaves_end,
	const u32 internals_start
) {
	u32 site = blockIdx.x * blockDim.x + threadIdx.x;
	if (site >= num_sites) {
		return;
	}
	u32 sub = threadIdx.y;

	__shared__ f64 s_likelihood[BLOCK_SIZE * 4];

	for (u32 i = 0; i < leaves_end; i++) {
		CALCULATE_LEAF_PROJECTION
	}

	for (u32 i = internals_start; i < num_updated_nodes; i++) {
		u32 left_edge = (nodes[i] - num_leaves) * 2;
		u32 right_edge = left_edge + 1;

		// thread-local likelihood
		f64 l_likelihood = projections[sidx(left_edge)] *
			projections[sidx(right_edge)];
		s_likelihood[threadIdx.x * 4 + sub] = l_likelihood;

		__syncthreads();
		// rebuild the likelihood from the 4 neighbouring threads
		auto likelihood = make_f64x4(
			s_likelihood[threadIdx.x * 4 + 0],
			s_likelihood[threadIdx.x * 4 + 1],
			s_likelihood[threadIdx.x * 4 + 2],
			s_likelihood[threadIdx.x * 4 + 3]
		);

		f64 projection = dot(
			transitions[i * 4 + sub],
			likelihood
		);

		projections[sidx(edges[i])] = projection;
	}
}

entrypoint __launch_bounds__(32)
void update_likelihoods(
	const u32 num_sites,
	const u32 num_leaves,

	f64x4* restrict projections,
	f64* restrict likelihoods,

	u32 root
) {
	SITE_PRELUDE

	u32 left_root_edge = (root - num_leaves) * 2;
	u32 right_root_edge = left_root_edge + 1;

	f64x4 likelihood = hadamard(
		projections[idx(left_root_edge)],
		projections[idx(right_root_edge)]
	);

	f64 sum = likelihood.x + likelihood.y + likelihood.z + likelihood.w;
	likelihoods[site] = log(sum);
}

entrypoint __launch_bounds__(128)
void copy_projections(
	const u32 num_sites,

	const f64x4* restrict src,
	f64x4* restrict dst,

	const u32* restrict edges
) {
	SITE_PRELUDE
	u32 i = blockIdx.y;

	u32 proj_idx = idx(edges[i]);
	dst[proj_idx] = src[proj_idx];
}
