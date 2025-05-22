#version 460

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly restrict buffer Length {
	uint num_rows;
};
layout(set = 0, binding = 1) restrict buffer Probabilities {
	dvec4 probabilities[];
};
layout(set = 0, binding = 2) restrict buffer Masks {
	uint masks[];
};

layout(set = 1, binding = 0) restrict readonly buffer Nodes {
	uint nodes[];
};
layout(set = 1, binding = 1) restrict readonly buffer Transitions {
	dmat4x4 transitions[];
};
layout(set = 1, binding = 2) restrict readonly buffer Children {
	uint children[];
};
layout(set = 1, binding = 3) restrict writeonly buffer Likelihoods {
	double likelihoods[];
};

void main() {
	uint idx = gl_GlobalInvocationID.x;
	uint offset = idx * num_rows;

	if (offset >= masks.length()) {
		return;
	}

	for (uint i = 0; i < nodes.length(); i++) {
		// the masks start at offset
		// the probabilities start at offset * 2

		uint left_child = children[i * 2];
		uint right_child = children[i * 2 + 1];

		uint left_idx = (offset + left_child) * 2 +
			masks[offset + left_child];
		uint right_idx = (offset + right_child) * 2 +
			masks[offset + right_child];

		dvec4 left = transitions[i * 2] * probabilities[left_idx];
		dvec4 right = transitions[i * 2 + 1] * probabilities[right_idx];

		uint node_idx = nodes[i] + offset;
		// flip the mask
		masks[node_idx] ^= 1;
		// write the new value
		probabilities[node_idx * 2 + masks[node_idx]] = left * right;
	}

	uint root = nodes[nodes.length() - 1];
	uint mask = masks[offset + root];
	dvec4 probability = probabilities[(offset + root) * 2 + mask];
	double sum = probability.x + probability.y + probability.z + probability.w;
	likelihoods[idx] = sum;
}
