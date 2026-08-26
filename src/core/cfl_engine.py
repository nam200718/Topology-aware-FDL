"""Clustered Federated Learning baseline (Sattler et al., TNNLS 2020).

Server-side iterative client clustering driven by gradient-update cosine
similarity. Each round, every client trains locally on its cluster model and
uploads the resulting update. The server aggregates per cluster and then runs
the recursive separation procedure of Algorithms 1-2: while any cluster holds
a pair of clients whose update cosine similarity falls below eps, the cluster
is split along the most dissimilar pair (hyperplane orthogonal to the update
difference through the less similar endpoint), and both sides are refined by
iteratively removing clients negatively correlated with the cluster-mean
update. Clients dropped by both refinements are attached to the side whose
mean update they correlate with more strongly (all clients remain assigned).
Following the paper's tuning recipe, eps starts high and is successively
decreased until the first separation is observed.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from src.core.centralized_engine import CentralizedEngine
from src.core.model import vector_to_model


class CFLEngine(CentralizedEngine):
    def __init__(self, config, topology, aggregator, device="cpu"):
        super().__init__(config, topology, aggregator, device)
        n = self.config.clients.num_clients
        self.cluster_members: Dict[int, List[int]] = {0: list(range(n))}
        self.cluster_models: Dict[int, torch.Tensor] = {0: self.server_weights.clone()}
        params = getattr(config.topology, "params", {}) or {}
        self.cfl_eps = float(params.get("cfl_eps_start", 0.1))
        self.cfl_eps_min = float(params.get("cfl_eps_min", 1e-8))
        self.cfl_decrease_factor = float(params.get("cfl_decrease_factor", 0.1))
        self.separation_observed = False

    # ------------------------------------------------------------------
    # Clustering primitives
    # ------------------------------------------------------------------

    def _cosine_matrix(self, members: List[int], deltas: Dict[int, torch.Tensor]) -> torch.Tensor:
        W = torch.stack([F.normalize(deltas[cid].float(), dim=0) for cid in members])
        return W @ W.T

    def _refine(self, members: List[int], deltas: Dict[int, torch.Tensor]) -> List[int]:
        """Iteratively remove clients with negative cosine to the cluster-mean
        update until all correlations are non-negative (Algorithm 1)."""
        members = list(members)
        while len(members) > 2:
            W = torch.stack([F.normalize(deltas[cid].float(), dim=0) for cid in members])
            mean_dir = F.normalize(W.mean(dim=0), dim=0)
            sims = W @ mean_dir
            bad = [k for k in range(len(members)) if sims[k].item() < 0.0]
            if not bad:
                break
            members = [members[k] for k in range(len(members)) if k not in set(bad)]
            if not members:
                members = []  # fully dissolved; caller reattaches
                break
        return members

    def _try_split(self, members: List[int], deltas: Dict[int, torch.Tensor]) -> Optional[List[List[int]]]:
        if len(members) < 3:
            return None
        members = sorted(members)
        C = self._cosine_matrix(members, deltas)
        Cm = C.clone()
        Cm.fill_diagonal_(2.0)
        min_val = float(Cm.min().item())
        if min_val >= self.cfl_eps:
            return None

        flat = int(Cm.argmin().item())
        i, j = divmod(flat, len(members))
        ci, cj = members[i], members[j]

        d = deltas[ci].float() - deltas[cj].float()
        dn = F.normalize(d, dim=0)
        W = torch.stack([deltas[cid].float() for cid in members])
        proj = (W - W[j:j + 1]) @ dn

        group_a = [members[k] for k in range(len(members)) if proj[k].item() > 0]
        group_b = [members[k] for k in range(len(members)) if proj[k].item() <= 0]
        if not group_a or not group_b:
            return None

        ref_a = self._refine(group_a, deltas)
        ref_b = self._refine(group_b, deltas)

        # Reattach clients removed by both refinements to the group whose
        # mean update correlates more strongly with theirs.
        dropped = [cid for cid in members if cid not in ref_a and cid not in ref_b]
        for cid in dropped:
            sims = {}
            for name, grp in (("a", ref_a), ("b", ref_b)):
                if grp:
                    M = torch.stack([F.normalize(deltas[x].float(), dim=0) for x in grp])
                    sims[name] = float((F.normalize(deltas[cid].float(), dim=0) @ M.mean(dim=0)).item())
                else:
                    sims[name] = -2.0
            (ref_a if sims["a"] >= sims["b"] else ref_b).append(cid)

        if not ref_a or not ref_b:
            return None
        return [sorted(ref_a), sorted(ref_b)]

    def _update_clusters(self, deltas: Dict[int, torch.Tensor]) -> None:
        stack = [(members, None) for members in self.cluster_members.values()]
        separated = False
        new_members: Dict[int, List[int]] = {}
        next_id = max(self.cluster_models.keys()) + 1

        while stack:
            members, parent_model = stack.pop()
            split = self._try_split(members, deltas)
            if split is None:
                new_members[next_id] = members
                # Children keep the model inherited from their parent; clusters
                # never split keep aggregating into the same slot.
                existing = None
                for k, mem in self.cluster_members.items():
                    if sorted(mem) == sorted(members):
                        existing = k
                        break
                if existing is not None:
                    new_members[existing] = new_members.pop(next_id)
                else:
                    self.cluster_models[next_id] = (
                        parent_model.clone() if parent_model is not None
                        else self.server_weights.clone())
                next_id += 1
            else:
                separated = True
                a, b = split
                for child_members in (a, b):
                    stack.append((child_members, self._model_for_members(members)))

        if separated:
            self.separation_observed = True
        else:
            self.cfl_eps = max(self.cfl_eps_min, self.cfl_eps * self.cfl_decrease_factor)

        self.cluster_members = new_members
        self.cluster_models = {
            k: v for k, v in self.cluster_models.items() if k in new_members
        }
        for k in new_members:
            self.cluster_models.setdefault(k, self.server_weights.clone())

    def _model_for_members(self, members: List[int]) -> torch.Tensor:
        for k, mem in self.cluster_members.items():
            if sorted(mem) == sorted(members):
                return self.cluster_models.get(k, self.server_weights)
        return self.server_weights

    def _cluster_of(self, client_id: int) -> int:
        for k, mem in self.cluster_members.items():
            if client_id in mem:
                return k
        return 0

    # ------------------------------------------------------------------
    # Round protocol
    # ------------------------------------------------------------------

    def run_round(self, round_num: int):
        target_clients = self.topology.get_server_connected_clients()

        # Download: each client receives its cluster model.
        received = {}
        for client_id in target_clients:
            k = self._cluster_of(client_id)
            self.cluster_models.setdefault(k, self.server_weights.clone())
            received[client_id] = self.cluster_models[k].clone()
            self.clients_state[client_id].weights = received[client_id].clone()

        current_lr = self.get_current_lr(round_num)
        updated_states = []
        for client_id in target_clients:
            state = self.clients_state[client_id].copy()
            client_ds = self.client_train_datasets[client_id]
            updated_state = self.updater.update(
                state=state,
                client_dataset=client_ds,
                config=self.config.clients,
                rng=self.local_rng,
                current_lr=current_lr,
            )
            self.clients_state[client_id] = updated_state
            updated_states.append(updated_state)

        deltas = {
            s.client_id: s.weights.float() - received[s.client_id].float().to(s.weights.device)
            for s in updated_states
        }

        # Per-cluster aggregation.
        expected = len(self.server_weights)
        valid = [s for s in updated_states if len(s.weights) == expected]
        for k, members in self.cluster_members.items():
            member_states = [s for s in valid if s.client_id in members]
            if member_states:
                self.cluster_models[k] = self.aggregator.aggregate(member_states)

        self._update_clusters(deltas)

        if not self.should_evaluate(round_num):
            self.metrics.log_round({
                "round": round_num,
                "participating_clients": len(target_clients),
                "total_clients_targeted": len(target_clients),
                "evaluated": False,
                "num_clusters": len(self.cluster_members),
                "cfl_eps": self.cfl_eps,
            })
            return

        ens_acc, ens_loss = self._evaluate_clusters()

        round_data = {
            "round": round_num,
            "test_accuracy": ens_acc,
            "test_loss": ens_loss,
            "participating_clients": len(target_clients),
            "total_clients_targeted": len(target_clients),
            "evaluated": True,
            "ensemble_test_accuracy": ens_acc,
            "ensemble_test_loss": ens_loss,
            "num_clusters": len(self.cluster_members),
            "cfl_eps": self.cfl_eps,
        }

        if round_num == self.config.num_rounds and getattr(self, "_last_per_client_accuracy", None):
            accs = sorted(self._last_per_client_accuracy.values())
            kk = max(1, int(torch.ceil(torch.tensor(0.1 * len(accs))).item()))
            round_data["bottom10_fairness"] = float(sum(accs[:kk]) / kk)
            round_data["per_client_accuracy"] = {
                str(cid): round(a, 4) for cid, a in sorted(self._last_per_client_accuracy.items())
            }

        self.metrics.log_round(round_data)

    def _evaluate_clusters(self):
        """Evaluate each client under its own cluster model on its local test
        partition (mean per-client accuracy, matching baseline metric format)."""
        from src.data.dataset import ClientDataset, get_fast_dataloader

        model = self.updater.global_model
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        per_client_stats = {}
        self._last_per_client_accuracy = {}

        current_cluster = None
        for client_id in range(self.config.clients.num_clients):
            k = self._cluster_of(client_id)
            if k != current_cluster:
                vector_to_model(self.cluster_models.get(
                    k, self.server_weights).to(self.device), model)
                model.eval()
                current_cluster = k

            client_test_ds = ClientDataset(self.test_dataset, self.client_test_indices[client_id])
            if len(client_test_ds) == 0:
                continue
            c_correct, c_total = 0, 0
            test_loader = get_fast_dataloader(client_test_ds,
                                              batch_size=min(len(client_test_ds), 1024),
                                              shuffle=False)
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    total_loss += loss.item()
                    hits = (logits.argmax(dim=1) == labels).sum().item()
                    total_correct += hits
                    total_samples += labels.size(0)
                    c_correct += hits
                    c_total += labels.size(0)
            if c_total > 0:
                per_client_stats[client_id] = (c_correct / c_total) * 100.0

        self._last_per_client_accuracy = per_client_stats
        if total_samples == 0:
            return 0.0, 0.0
        return 100 * total_correct / total_samples, total_loss / total_samples
