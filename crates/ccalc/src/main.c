#include <assert.h>
#include <math.h>
#include <stdio.h>

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

typedef signed char i8;
typedef signed short i16;
typedef signed int i32;
typedef signed long long i64;

typedef float f32;
typedef double f64;
typedef struct {
	f64 inner[4];
} f64x4;

void calculate_leaf(
	u32 num_patterns,
	const f64* restrict transition,
	f64* restrict projections,
	const u8* restrict samples
) {
	f64 m_00 = transition[ 0];
	f64 m_01 = transition[ 1];
	f64 m_02 = transition[ 2];
	f64 m_03 = transition[ 3];
	f64 m_10 = transition[ 4];
	f64 m_11 = transition[ 5];
	f64 m_12 = transition[ 6];
	f64 m_13 = transition[ 7];
	f64 m_20 = transition[ 8];
	f64 m_21 = transition[ 9];
	f64 m_22 = transition[10];
	f64 m_23 = transition[11];
	f64 m_30 = transition[12];
	f64 m_31 = transition[13];
	f64 m_32 = transition[14];
	f64 m_33 = transition[15];

	u32 end = num_patterns * 4;

	/*
	#pragma omp parallel for default(none) \
		firstprivate(projections, samples, end) \
		firstprivate(m_00, m_01, m_02, m_03, m_10, m_11, m_12, m_13, m_20, m_21, m_22, m_23, m_30, m_31, m_32, m_33) \
		schedule(static)
	*/
	for (u32 u = 0; u < end; u += 4) {
		u8 sample = samples[u / 4];
		if (sample == 0b0001) {
			projections[u + 0] = m_00;
			projections[u + 1] = m_10;
			projections[u + 2] = m_20;
			projections[u + 3] = m_30;
		} else if (sample == 0b0010) {
			projections[u + 0] = m_01;
			projections[u + 1] = m_11;
			projections[u + 2] = m_21;
			projections[u + 3] = m_31;
		} else if (sample == 0b0100) {
			projections[u + 0] = m_02;
			projections[u + 1] = m_12;
			projections[u + 2] = m_22;
			projections[u + 3] = m_32;
		} else if (sample == 0b1000) {
			projections[u + 0] = m_03;
			projections[u + 1] = m_13;
			projections[u + 2] = m_23;
			projections[u + 3] = m_33;
		} else {
			projections[u + 0] = 1.0;
			projections[u + 1] = 1.0;
			projections[u + 2] = 1.0;
			projections[u + 3] = 1.0;
		}
	}
}

void calculate_projection(
	u32 num_patterns,
	const f64* restrict projections_left,
	const f64* restrict projections_right,
	f64* restrict projections_node,
	const f64* restrict transition
) {
	f64 m_00 = transition[ 0];
	f64 m_01 = transition[ 1];
	f64 m_02 = transition[ 2];
	f64 m_03 = transition[ 3];
	f64 m_10 = transition[ 4];
	f64 m_11 = transition[ 5];
	f64 m_12 = transition[ 6];
	f64 m_13 = transition[ 7];
	f64 m_20 = transition[ 8];
	f64 m_21 = transition[ 9];
	f64 m_22 = transition[10];
	f64 m_23 = transition[11];
	f64 m_30 = transition[12];
	f64 m_31 = transition[13];
	f64 m_32 = transition[14];
	f64 m_33 = transition[15];

	u32 end = num_patterns * 4;
	/*
	#pragma omp parallel for default(none) \
		firstprivate(projections_left, projections_right, projections_node, end) \
		firstprivate(m_00, m_01, m_02, m_03, m_10, m_11, m_12, m_13, m_20, m_21, m_22, m_23, m_30, m_31, m_32, m_33) \
		schedule(static)
	*/
	for (u32 u = 0; u < end; u += 4) {
		f64 p0 = projections_left[u + 0] * projections_right[u + 0];
		f64 p1 = projections_left[u + 1] * projections_right[u + 1];
		f64 p2 = projections_left[u + 2] * projections_right[u + 2];
		f64 p3 = projections_left[u + 3] * projections_right[u + 3];

		projections_node[u + 0] = p0 * m_00 + p1 * m_01 + p2 * m_02 + p3 * m_03;
		projections_node[u + 1] = p0 * m_10 + p1 * m_11 + p2 * m_12 + p3 * m_13;
		projections_node[u + 2] = p0 * m_20 + p1 * m_21 + p2 * m_22 + p3 * m_23;
		projections_node[u + 3] = p0 * m_30 + p1 * m_31 + p2 * m_32 + p3 * m_33;
	}
}

#define P_OFFSET(idx) \
	(((idx) * 2 + selectors[(idx)]) * num_patterns) * 4

void propose(
	const u32* restrict nodes,
	u32 nodes_len,
	const u32* restrict children,
	const f64* restrict transitions,
	u32 leaves_end,
	f64x4 frequencies,

	u32 num_patterns,
	const u8* restrict samples,
	f64* restrict projections,
	u8* restrict selectors,
	f64* restrict likelihoods
) {
	for (u32 i = 0; i < leaves_end; i += 1) {
		u32 leaf = nodes[i];
		selectors[leaf] ^= 1;

		const f64* transition = transitions + i * 16;

		f64* projections_leaf = projections + P_OFFSET(leaf);

		const u8* samples_leaf = samples + leaf * num_patterns;

		calculate_leaf(
			num_patterns,
			transition,
			projections_leaf,
			samples_leaf
		);
	}

	for (u32 i = leaves_end; i < nodes_len - 1; i += 1) {
		u32 node = nodes[i];
		selectors[node] ^= 1;

		const f64* transition = transitions + i * 16;
		u32 left = children[(i - leaves_end) * 2];
		u32 right = children[(i - leaves_end) * 2 + 1];

		f64* projections_left = projections + P_OFFSET(left);
		f64* projections_right = projections + P_OFFSET(right);
		f64* projections_node = projections + P_OFFSET(node);

		calculate_projection(
			num_patterns,
			projections_left,
			projections_right,
			projections_node,
			transition
		);
	}

	u32 root_left = children[(nodes_len - 1 - leaves_end) * 2];
	u32 root_right = children[(nodes_len - 1 - leaves_end) * 2 + 1];

	f64* projections_left = projections + P_OFFSET(root_left);
	f64* projections_right = projections + P_OFFSET(root_right);

	u32 u = 0;
	for (u32 i = 0; i < num_patterns; i += 1) {
		f64 p0 = projections_left[u + 0] * projections_right[u + 0];
		f64 p1 = projections_left[u + 1] * projections_right[u + 1];
		f64 p2 = projections_left[u + 2] * projections_right[u + 2];
		f64 p3 = projections_left[u + 3] * projections_right[u + 3];

		likelihoods[i] = log(
			p0 * frequencies.inner[0] +
			p1 * frequencies.inner[1] +
			p2 * frequencies.inner[2] +
			p3 * frequencies.inner[3]
		);

		u += 4;
	}
}
