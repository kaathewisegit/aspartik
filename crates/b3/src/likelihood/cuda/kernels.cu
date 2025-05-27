typedef unsigned char byte;
typedef unsigned int uint;

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

__device__ double4 mat_aplly(
	const double4* __restrict__ matrix,
	const double4 vector
) {
	return make_double4(
		dot(matrix[0], vector),
		dot(matrix[1], vector),
		dot(matrix[2], vector),
		dot(matrix[3], vector)
	);
}

#define idx(edge) \
	((edge) * num_sites + site)

extern "C" __global__ void propose(
	const uint num_edges,
	const uint num_sites,

	const double4* __restrict__ leaves,
	double4* __restrict__ projections,

	const uint num_updated_nodes,
	const uint* __restrict__ nodes,
	const uint* __restrict__ edges,
	const double4* __restrict__ transitions,
	const uint cutoff,
	const uint root,
	double* __restrict__ likelihoods
) {
	uint site = blockIdx.x * blockDim.x + threadIdx.x;
	if (site >= num_sites) {
		return;
	}
	uint num_leaves = (num_edges / 2) + 1;

	for (uint i = 0; i < cutoff; i++) {
		double4 projection = mat_aplly(
			transitions + i * 4,
			leaves[nodes[i] * num_sites + site]
		);

		projections[idx(edges[i])] = projection;
	}

	for (uint i = cutoff; i < num_updated_nodes; i++) {
		uint left_edge = (nodes[i] - num_leaves) * 2;
		uint right_edge = left_edge + 1;

		double4 likelihood = hadamard(
			projections[idx(left_edge)],
			projections[idx(right_edge)]
		);

		double4 projection = mat_aplly(
			transitions + i * 4,
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
	likelihoods[site] = sum;
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
