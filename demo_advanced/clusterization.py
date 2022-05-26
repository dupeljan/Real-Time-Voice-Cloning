import numpy as np
import json

from enum import Enum
from pathlib import Path
from scipy.spatial.distance import cdist
from typing import Callable


class Strategy(Enum):
    linear = 0
    adaptive = 1


class SpeakerMixer:
    def __init__(self, cluster_path: Path, strategy: Strategy = Strategy.linear,
                 mix_c: float = 0, mix_c_file: Path = None, mix_c_adaptive=15):
        self.data_root = cluster_path
        self.strategy = strategy
        self._speakers_names = []
        self.mix_with_most_similar = self._mix_factory()
        self._mix_c = mix_c
        if mix_c_file is not None:
            with open(mix_c_file, 'r') as f:
                self._mix_c = json.load(f)
        self._mix_c_adaptive = mix_c_adaptive

        if self.data_root:
            embeddings = []
            for fname in self.data_root.glob('*.npy'):
                self._speakers_names.append(fname.name)
                embeddings.append(np.load(fname))
            self._embeddings = np.stack(embeddings)

    @property
    def mix_coef(self):
        return self._mix_c

    def get_configuration(self, speaker) -> str:
        retval = {'mode': self.strategy.name}
        if self.strategy == Strategy.adaptive:
            retval['mix_c'] = self._mix_c_adaptive
            return retval
        retval['mix_c'] = self._mix_c[speaker] if isinstance(self._mix_c, dict) else self._mix_c
        return retval

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

    def _mix_factory(self) -> Callable[[np.array, str], np.array]:
        if not self.data_root:
            return lambda x, y: x
        if self.strategy == Strategy.linear:
            def mix(inp_emb, speaker: str = None):
                most_similar_emb, _, _ = self.get_most_similar_emb(inp_emb)
                mix_c = self._mix_c[speaker] if isinstance(self._mix_c, dict) else self._mix_c
                return inp_emb + mix_c * (most_similar_emb - inp_emb)
            return mix
        if self.strategy == Strategy.adaptive:
            def mix(inp_emb, speaker: str = None):
                most_similar_emb, _, score = self.get_most_similar_emb(inp_emb)
                c = max(min(self._mix_c_adaptive * score, 1), 0)
                return inp_emb + c * (most_similar_emb - inp_emb)
            return mix