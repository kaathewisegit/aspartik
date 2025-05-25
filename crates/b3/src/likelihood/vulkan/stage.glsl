#version 460

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) restrict uniform PushConstants {
	uint num_nodes;
	uint num_sites;
};

layout(set = 0, binding = 0) restrict writeonly buffer Probabilities {
	dvec4 probabilities[];
};
layout(set = 0, binding = 1) restrict writeonly buffer Masks {
	uint masks[];
};

layout(set = 3, binding = 0) restrict readonly buffer StagingProbabilities {
	dvec4 stage_probabilities[];
};
layout(set = 3, binding = 1) restrict readonly buffer StagingMasks {
	uint stage_masks[];
};

void main() {
	uint idx = gl_GlobalInvocationID.x;
	uint offset = idx * num_nodes;

	if (offset >= masks.length()) {
		return;
	}

	for (uint i = 0; i < num_nodes; i++) {
		// the masks start at offset
		// the probabilities start at offset * 2

		uint prob_idx = (offset + i) * 2;
		probabilities[prob_idx] = stage_probabilities[prob_idx];
		probabilities[prob_idx + 1] = stage_probabilities[prob_idx + 1];

		uint mask_idx = offset + i;
		masks[mask_idx] = stage_masks[mask_idx];
	}
}
