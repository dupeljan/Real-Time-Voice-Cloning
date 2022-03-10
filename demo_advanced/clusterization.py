import numpy as np

from enum import Enum
from pathlib import Path
from scipy.spatial.distance import cdist
from typing import Callable


class Strategy(Enum):
    linear = 0
    adaptive = 1


class SpeakerMixer:
    def __init__(self, cluster_path: Path, strategy: Strategy = Strategy.linear,
                 mix_c: float = 0, mix_c_adaptive=15):
        self.data_root = cluster_path
        if cluster_path is None:
            return
        self.strategy = strategy
        self._speakers_names = []
        embeddings = []
        for fname in cluster_path.glob('*.npy'):
            self._speakers_names.append(fname.name)
            embeddings.append(np.load(fname))
        self._embeddings = np.stack(embeddings)
        self.mix_with_most_similar = self._mix_factory()
        self._mix_c = mix_c
        self._mix_c_adaptive = mix_c_adaptive

    def get_most_similar_emb(self, inp_emb: np.array) -> (np.array, str, float):
        """
        Return most similar embeddings from given clusterisation.

        :inp_emb: Input speaker embedding.
        :return: (most similar to input speaker speaker embedding,
            most similar to input speaker speaker name, cosine score of input and most similar embeddings)
        """
        if not self.data_root:
            return inp_emb, None, 0.

        assert len(inp_emb.shape) == 1
        dist = cdist(inp_emb.reshape(1, -1), self._embeddings, 'cosine')
        idx = np.argmin(dist)
        return self._embeddings[idx, :], self._speakers_names[idx], dist[0, idx]

    def _mix_factory(self) -> Callable[[np.array], np.array]:
        if not self.data_root:
            return lambda x: x
        if self.strategy == Strategy.linear:
            def mix(inp_emb):
                most_similar_emb, _, _ = self.get_most_similar_emb(inp_emb)
                return inp_emb + self._mix_c * (most_similar_emb - inp_emb)
            return mix
        if self.strategy == Strategy.adaptive:
            def mix(inp_emb):
                most_similar_emb, _, score = self.get_most_similar_emb(inp_emb)
                c = max(min(self._mix_c_adaptive * score, 1), 0)
                return inp_emb + c * (most_similar_emb - inp_emb)
            return mix