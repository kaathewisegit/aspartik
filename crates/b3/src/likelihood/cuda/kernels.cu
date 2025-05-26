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
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_sites) {
		return;
	}
	uint offset = idx * num_edges;
	uint num_leaves = (num_edges / 2) + 1;
	uint leaves_offset = idx * num_leaves;

	for (uint i = 0; i < cutoff; i++) {
		double4 projection = mat_aplly(
			transitions + i * 4,
			leaves[leaves_offset + nodes[i]]
		);

		uint proj_idx = edges[i] + offset;
		masks[proj_idx] ^= 1;
		projections[proj_idx * 2 + masks[proj_idx]] = projection;
	}

	for (uint i = cutoff; i < num_updated_nodes; i++) {
		uint left_edge = (nodes[i] - num_leaves) * 2;
		uint right_edge = left_edge + 1;

		uint left_idx = (offset + left_edge) * 2 +
			masks[offset + left_edge];
		uint right_idx = (offset + right_edge) * 2 +
			masks[offset + right_edge];

		double4 likelihood = hadamard(
			projections[left_idx],
			projections[right_idx]
		);

		double4 projection = mat_aplly(
			transitions + i * 4,
			likelihood
		);

		uint proj_idx = edges[i] + offset;
		masks[proj_idx] ^= 1;
		projections[proj_idx * 2 + masks[proj_idx]] = projection;
	}

	uint root_left = (root - num_leaves) * 2;
	uint root_right = root_left + 1;
	uint left_idx = (offset + root_left) * 2 +
		masks[offset + root_left];
	uint right_idx = (offset + root_right) * 2 +
		masks[offset + root_right];

	double4 likelihood = hadamard(
		projections[left_idx],
		projections[right_idx]
	);

	double sum = likelihood.x + likelihood.y + likelihood.z + likelihood.w;
	likelihoods[idx] = sum;
}

extern "C"  __global__ void reject(
	const uint num_edges,
	const uint num_sites,
	byte* __restrict__ masks,
	const uint num_updated_nodes,
	const uint* __restrict__ edges
) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_sites) {
		return;
	}
	uint offset = idx * num_edges;

	for (uint i = 0; i < num_updated_nodes; i++) {
		masks[offset + edges[i]] ^= 1;
	}
}
