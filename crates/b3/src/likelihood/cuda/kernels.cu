typedef unsigned char byte;
typedef unsigned int uint;

typedef struct {
	double4 a, c, g, t;
} Transition;

__device__ double dot(const double4 a, const double4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ double4 hadamard(const double4 a, const double4 b) {
	return make_double4(
		a.x * b.x,
		a.y * b.y,
		a.z * b.z,
		a.w * b.w
	);
}

__device__ double4 apply(
	const Transition transition,
	const double4 vector
) {
	return make_double4(
		dot(transition.a, vector),
		dot(transition.c, vector),
		dot(transition.g, vector),
		dot(transition.t, vector)
	);
}

extern "C" __global__ void update_leaves(
	const uint num_sites,

	const uint* __restrict__ edges,
	const uint* __restrict__ nodes,
	const Transition* __restrict__ transitions,

	const double4* __restrict__ leaves,
	double4* __restrict__ projections
) {
	uint site = threadIdx.x;
	uint i = blockIdx.x;

	uint edge = edges[i];
	uint leaf_idx = nodes[i];
	Transition transition = transitions[i];

	uint leaf_offset = leaf_idx * num_sites;
	uint edge_offset = edge * num_sites;

	double4 leaf = leaves[leaf_offset + site];
	double4 projection = apply(transition, leaf);
	projections[edge_offset + site] = projection;
}

#define idx(edge) \
	((edge) * num_sites + site)

extern "C" __global__ void propose(
	const uint num_sites,
	const uint num_leaves,

	const double4* __restrict__ leaves,
	double4* __restrict__ projections,

	const uint num_updated_nodes,
	const uint* __restrict__ nodes,
	const uint* __restrict__ edges,
	const Transition* __restrict__ transitions,
	const uint cutoff,
	const uint root,
	double* __restrict__ likelihoods
) {
	uint site = blockIdx.x * blockDim.x + threadIdx.x;
	if (site >= num_sites) {
		return;
	}

	if (cutoff > 10) {
		for (uint i = 0; i < cutoff; i++) {
			double4 projection = apply(
				transitions[i],
				leaves[nodes[i] * num_sites + site]
			);

			projections[idx(edges[i])] = projection;
		}
	}

	for (uint i = cutoff; i < num_updated_nodes; i++) {
		uint left_edge = (nodes[i] - num_leaves) * 2;
		uint right_edge = left_edge + 1;

		double4 likelihood = hadamard(
			projections[idx(left_edge)],
			projections[idx(right_edge)]
		);

		double4 projection = apply(
			transitions[i],
			likelihood
		);

		projections[idx(edges[i])] = projection;
	}

	uint left_root_edge = (root - num_leaves) * 2;
	uint right_root_edge = left_root_edge + 1;

	double4 likelihood = hadamard(
		projections[idx(left_root_edge)],
		projections[idx(right_root_edge)]
	);

	double sum = likelihood.x + likelihood.y + likelihood.z + likelihood.w;
	likelihoods[site] = log(sum);
}

extern "C"  __global__ void reject(
	const uint num_sites,

	double4* __restrict__ projections,
	const double4* __restrict__ projections_backup,

	const uint num_updated_nodes,
	const uint* __restrict__ edges
) {
	uint site = blockIdx.x * blockDim.x + threadIdx.x;
	if (site >= num_sites) {
		return;
	}

	for (uint i = 0; i < num_updated_nodes; i++) {
		uint proj_idx = idx(edges[i]);
		projections[proj_idx] = projections_backup[proj_idx];
	}
}

extern "C"  __global__ void accept(
	const uint num_sites,

	const double4* __restrict__ projections,
	double4* __restrict__ projections_backup,

	const uint num_updated_nodes,
	const uint* __restrict__ edges
) {
	uint site = blockIdx.x * blockDim.x + threadIdx.x;
	if (site >= num_sites) {
		return;
	}

	for (uint i = 0; i < num_updated_nodes; i++) {
		uint proj_idx = idx(edges[i]);
		projections_backup[proj_idx] = projections[proj_idx];
	}
}
