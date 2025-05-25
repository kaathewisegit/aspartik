#version 460

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) restrict uniform PushConstants {
	uint num_nodes;
	uint num_sites;
};

layout(set = 0, binding = 0) restrict readonly buffer Probabilities {
	dvec4 probabilities[];
};
layout(set = 0, binding = 1) restrict buffer Masks {
	uint masks[];
};

layout(set = 1, binding = 0) restrict readonly buffer Nodes {
	uint nodes[];
};
layout(set = 1, binding = 1) restrict uniform NodesLength {
	uint nodes_length;
};

void main() {
	uint idx = gl_GlobalInvocationID.x;
	uint offset = idx * num_nodes;

	if (offset >= masks.length()) {
		return;
	}
	uint len = nodes_length;

	for (uint i = 0; i < len; i++) {
		// the masks start at offset
		// the probabilities start at offset * 2

		// flip the mask
		masks[offset + nodes[i]] ^= 1;
	}
}
