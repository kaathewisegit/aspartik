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

// Update partial likelihoods for edges which go to leaves
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

// e^-40
constexpr f64 CUTOFF = 0.000000000000000004248354255291589;
// e^40
constexpr f64 MULT = 235385266837020000.0;

entrypoint __launch_bounds__(BLOCK_SIZE)
void propose(
	const u32 num_sites,
	const u32 num_leaves,

	const f64x4* restrict leaves,
	f64* restrict projections,
	u8* restrict scales,
	u32* scale_sums,

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

	u32 scale_sum = scale_sums[site];

	for (u32 i = internals_start; i < num_updated_nodes; i++) {
		u32 left_edge = (nodes[i] - num_leaves) * 2;
		u32 right_edge = left_edge + 1;
		u32 this_edge = edges[i];

		// thread-local likelihood
		f64 l_likelihood = projections[sidx(left_edge)] *
			projections[sidx(right_edge)];
		s_likelihood[threadIdx.x * 4 + sub] = l_likelihood;

		__syncthreads();

		if (sub == 0) {
			u32 scale_idx = idx(this_edge);
			u32 old_scale = scales[scale_idx];

			if (
				s_likelihood[threadIdx.x * 4 + 0] < CUTOFF
				&& s_likelihood[threadIdx.x * 4 + 1] < CUTOFF
				&& s_likelihood[threadIdx.x * 4 + 2] < CUTOFF
				&& s_likelihood[threadIdx.x * 4 + 3] < CUTOFF
			) {
				s_likelihood[threadIdx.x * 4 + 0] *= MULT;
				s_likelihood[threadIdx.x * 4 + 1] *= MULT;
				s_likelihood[threadIdx.x * 4 + 2] *= MULT;
				s_likelihood[threadIdx.x * 4 + 3] *= MULT;

				if (old_scale == 0) {
					scale_sum += 40;
					scales[scale_idx] = 1;
				}
			} else {
				if (old_scale == 1) {
					scale_sum -= 40;
					scales[scale_idx] = 0;
				}
			}
		}

		__syncthreads();

		// rebuild the likelihood from the 4 neighboring threads
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

		projections[sidx(this_edge)] = projection;
	}

	if (sub == 0) scale_sums[site] = scale_sum;
}

entrypoint __launch_bounds__(32)
void update_likelihoods(
	const u32 num_sites,
	const u32 num_leaves,

	const f64x4* restrict projections,
	f64* restrict likelihoods,

	const u32* restrict edges,

	u32 root
) {
	SITE_PRELUDE

	u32 num_edges = (num_leaves - 1) * 2;

	u32 left_root_edge = (root - num_leaves) * 2;
	u32 right_root_edge = left_root_edge + 1;

	f64x4 likelihood = hadamard(
		projections[idx(left_root_edge)],
		projections[idx(right_root_edge)]
	);

	f64 sum = likelihood.x + likelihood.y + likelihood.z + likelihood.w;
	likelihoods[site] = log(sum);
}

// Updates the backups or resets the working arrays
//
// TODO: it might make sense to separate it into two kernels: one for
// projections and one for scales.  I can't use the default memcpy because it
// needs to sample by edges.
entrypoint __launch_bounds__(128)
void copy_projections(
	const u32 num_sites,

	const f64x4* restrict p_src,
	f64x4* restrict p_dst,

	const u8* restrict s_src,
	u8* restrict s_dst,

	const u32* restrict edges
) {
	SITE_PRELUDE
	u32 i = blockIdx.y;

	u32 proj_idx = idx(edges[i]);
	p_dst[proj_idx] = p_src[proj_idx];
	s_dst[proj_idx] = s_src[proj_idx];
}
