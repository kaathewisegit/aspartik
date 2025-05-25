typedef unsigned int u32;

extern "C" __global__ void reject(
	u32 num_nodes,
	char* masks,
	u32 num_updated_nodes,
	u32 *updated_nodes
) {
	u32 idx = threadIdx.x;
	if (idx >= num_nodes) {
		return;
	}
	u32 offset = idx * num_nodes;

	for (u32 i = 0; i < num_updated_nodes; i++) {
		masks[offset + updated_nodes[i]] ^= 1;
	}
}
