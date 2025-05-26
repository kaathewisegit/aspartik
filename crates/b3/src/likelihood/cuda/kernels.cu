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

extern "C" __global__ void propose(
	const uint num_edges,
	const uint num_sites,
	double4* __restrict__ leaves,
	byte* __restrict__ masks,
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

		uint node_idx = edges[i] * num_sites + site;
		masks[node_idx] ^= 1;
		projections[node_idx * 2 + masks[node_idx]] = projection;
	}

	for (uint i = cutoff; i < num_updated_nodes; i++) {
		uint left_edge = (nodes[i] - num_leaves) * 2;
		uint right_edge = left_edge + 1;

		uint left_idx = (left_edge * num_sites + site) * 2 +
			masks[left_edge * num_sites + site];
		uint right_idx = (right_edge * num_sites + site) * 2 +
			masks[right_edge * num_sites + site];

		double4 likelihood = hadamard(
			projections[left_idx],
			projections[right_idx]
		);

		double4 projection = mat_aplly(
			transitions + i * 4,
			likelihood
		);

		uint node_idx = edges[i] * num_sites + site;
		masks[node_idx] ^= 1;
		projections[node_idx * 2 + masks[node_idx]] = projection;
	}

	uint left_root_edge = (root - num_leaves) * 2;
	uint right_root_edge = left_root_edge + 1;
	uint left_idx = (left_root_edge * num_sites + site) * 2 +
		masks[left_root_edge * num_sites + site];
	uint right_idx = (right_root_edge * num_sites + site) * 2 +
		masks[right_root_edge * num_sites + site];

	double4 likelihood = hadamard(
		projections[left_idx],
		projections[right_idx]
	);

	double sum = likelihood.x + likelihood.y + likelihood.z + likelihood.w;
	likelihoods[site] = sum;
}

extern "C"  __global__ void reject(
	const uint num_edges,
	const uint num_sites,
	byte* __restrict__ masks,
	const uint num_updated_nodes,
	const uint* __restrict__ edges
) {
	uint site = blockIdx.x * blockDim.x + threadIdx.x;
	if (site >= num_sites) {
		return;
	}

	for (uint i = 0; i < num_updated_nodes; i++) {
		masks[edges[i] * num_sites + site] ^= 1;
	}
}
