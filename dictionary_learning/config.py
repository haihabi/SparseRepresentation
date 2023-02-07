from dataclasses import dataclass

from dictionary_learning.dictionary_initialization import InitializationMethod
from dictionary_learning.sparse_recovery import SparseRecoveryMethod


@dataclass
class KSVDConfig:
    n_iterations: int
    k_sparse: int
    num_atoms: int
    in_sparse_recovery: SparseRecoveryMethod = SparseRecoveryMethod.MP
    initialization_method: InitializationMethod = InitializationMethod.DCT
    initial_dictionary = None
    etol: float = 1e-10
    approx: bool = False

    @staticmethod
    def get_default_config():
        return KSVDConfig(15, 0, 128)
