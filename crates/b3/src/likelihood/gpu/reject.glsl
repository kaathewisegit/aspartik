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

void main() {
	uint idx = gl_GlobalInvocationID.x;
	uint offset = idx * num_rows;

	if (offset >= masks.length()) {
		return;
	}

	for (uint i = 0; i < nodes.length(); i++) {
		// the masks start at offset
		// the probabilities start at offset * 2

		// flip the mask
		masks[offset + nodes[i]] ^= 1;
	}
}
