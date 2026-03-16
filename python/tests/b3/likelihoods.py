from aspartik.b3 import Clock
from aspartik.b3.likelihoods import CPU4Likelihood
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.substitutions import HKY
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG


def test_cpu(rng: RNG):
    msa = read_msa_from_fasta("data/alignments/apes.fasta")

    tree = Tree(msa.sequence_names(), rng)

    likelihood = CPU4Likelihood(
        msa=msa,
        substitution=HKY(RealVector(0.1, 0.2, 0.3, 0.4), Real(2.0)),
        clock=Clock.Strict(Real(1.0)),
        tree=tree,
    )

    assert likelihood.num_patterns() == 69

    for _ in range(1000):
        match rng.random_int(0, 3):
            case 0:
                likelihood.likelihood()
            case 1:
                likelihood.accept()
            case 2:
                likelihood.reject()
