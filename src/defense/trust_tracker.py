from typing import Dict, List, Optional
from collections import defaultdict


class TrustTracker:
    """
    Track trust score of each client through each round.
    Data is stored in the following structure:
      history[round_num][cluster_head_id] = {client_id: trust_score}
    """

    def __init__(self):
        # {round_num: {head_id: {client_id: trust_score}}}
        self.history: Dict[int, Dict[int, Dict[int, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )

    def log(
        self,
        round_num: int,
        head_id: int,
        trust_scores: Dict[int, float],
    ):
        """
        Log trust scores for 1 cluster in 1 round.

        Args:
            round_num: Current round number.
            head_id: ID of cluster head (e.g., -2, -3, -4).
            trust_scores: Dict {client_id: trust_score} from aggregator.
        """
        self.history[round_num][head_id] = dict(trust_scores)

    def get_client_history(self, client_id: int) -> Dict[int, float]:
        """
        Get trust score history of 1 client through all rounds.
        Returns: {round_num: trust_score}
        """
        result = {}
        for round_num, clusters in sorted(self.history.items()):
            for head_id, scores in clusters.items():
                if client_id in scores:
                    result[round_num] = scores[client_id]
        return result

    def get_round_scores(self, round_num: int) -> Dict[int, float]:
        """
        Get trust scores of all clients in 1 round (combine all clusters).
        Returns: {client_id: trust_score}
        """
        result = {}
        if round_num in self.history:
            for head_id, scores in self.history[round_num].items():
                result.update(scores)
        return result

    def get_all_client_ids(self) -> List[int]:
        """Get list of all client IDs that have been recorded."""
        all_ids = set()
        for round_data in self.history.values():
            for scores in round_data.values():
                all_ids.update(scores.keys())
        return sorted(all_ids)

    def get_all_rounds(self) -> List[int]:
        """Get list of all round numbers that have been recorded."""
        return sorted(self.history.keys())

    def to_matrix(self) -> tuple:
        """
        Convert history to numpy-friendly matrix.
        Returns: (client_ids, rounds, matrix) 
          - matrix[i][j] = trust score của client_ids[i] tại rounds[j]
        """
        client_ids = self.get_all_client_ids()
        rounds = self.get_all_rounds()

        matrix = []
        for cid in client_ids:
            row = []
            for r in rounds:
                scores = self.get_round_scores(r)
                row.append(scores.get(cid, float("nan")))
            matrix.append(row)

        return client_ids, rounds, matrix
