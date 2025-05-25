typedef unsigned char byte;
typedef unsigned int uint;

__device__ double dot(double4 a, double4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ double4 hadamard(double4 a, double4 b) {
	return make_double4(
		a.x * b.x,
		a.y * b.y,
		a.z * b.z,
		a.w * b.w
	);
}

__device__ double4 mul(
	const uint index,
	const double4* transitions,
	const double4 vector
) {
	return make_double4(
		dot(transitions[index + 0], vector),
		dot(transitions[index + 1], vector),
		dot(transitions[index + 2], vector),
		dot(transitions[index + 3], vector)
	);
}

extern "C" __global__ void propose(
	const uint num_nodes,
	const uint num_sites,
	byte* masks,
	double4* probabilities,

	const uint num_updated_nodes,
	const uint* updated_nodes,
	const uint* children,
	const double4* transitions,
	double* likelihoods
) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_sites) {
		return;
	}
	uint offset = idx * num_nodes;

	for (uint i = 0; i < num_updated_nodes; i++) {
		uint left_child = children[i * 2];
		uint right_child = children[i * 2 + 1];

		uint left_idx = (offset + left_child) * 2 +
			masks[offset + left_child];
		uint right_idx = (offset + right_child) * 2 +
			masks[offset + right_child];

		double4 left = mul(
			// 2i is the transition, x4 for four rows
			(i * 2) * 4,
			transitions,
			probabilities[left_idx]
		);
		double4 right = mul(
			(i * 2 + 1) * 4,
			transitions,
			probabilities[right_idx]
		);

		uint node_idx = updated_nodes[i] + offset;
		masks[node_idx] ^= 1;
		uint prob_idx = node_idx * 2 + masks[node_idx];
		probabilities[prob_idx] = hadamard(left, right);
	}

	uint root = updated_nodes[num_updated_nodes - 1];
	uint mask = masks[offset + root];
	double4 probability = probabilities[(offset + root) * 2 + mask];
	double sum = probability.x + probability.y + probability.z + probability.w;
	likelihoods[idx] = sum;
}

extern "C"  __global__ void reject(
	const uint num_nodes,
	const uint num_sites,
	byte* masks,
	const uint num_updated_nodes,
	const uint* updated_nodes
) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_sites) {
		return;
	}
	uint offset = idx * num_nodes;

	for (uint i = 0; i < num_updated_nodes; i++) {
		masks[offset + updated_nodes[i]] ^= 1;
	}
}
