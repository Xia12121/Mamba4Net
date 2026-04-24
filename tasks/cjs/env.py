"""Decima-style cluster job scheduling environment.

Jobs are DAGs of stages; each stage is a set of tasks with a fixed
per-task duration and memory footprint. At each decision step the agent
picks (frontier_stage, num_executors) and the simulator rolls time
forward until the next scheduling event. Reward = -ΔJCT so minimising
expected cumulative reward minimises JCT.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch


@dataclass
class Stage:
    stage_id: int
    job_id: int
    num_tasks: int
    task_duration: float
    parents: list[int]
    finished_tasks: int = 0
    running_tasks: int = 0
    # Populated at runtime
    start_time: float | None = None
    end_time: float | None = None

    @property
    def done(self) -> bool:
        return self.finished_tasks >= self.num_tasks


@dataclass
class SchedulingState:
    """Node features + DAG edges for the frontier & in-flight stages.

    ``node_feat`` rows align with ``node_ids`` so the agent can select a
    stage by index. The DAG includes parents even if completed (they
    contribute zero features but preserve topology)."""

    node_feat: torch.Tensor          # (N, F)
    edge_index: torch.Tensor         # (2, E)
    node_ids: list[int]              # stage indices
    frontier_mask: torch.Tensor      # (N,) bool - schedulable stages
    global_feat: torch.Tensor        # (G,)


EXECUTOR_BUCKETS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256]


class ClusterSchedulingEnv:
    def __init__(
        self,
        traces_dir: str | Path | None = None,
        num_executors: int = 50,
        num_jobs_per_episode: int = 20,
        seed: int = 0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.num_executors = num_executors
        self.num_jobs = num_jobs_per_episode
        self.traces_dir = Path(traces_dir) if traces_dir else None
        self.executor_buckets = EXECUTOR_BUCKETS

    # ------------------------------------------------------------------
    def _sample_job(self, job_id: int, start_time: float) -> list[Stage]:
        """Synthetic TPC-H-style DAG sampler (a tree with 4-12 stages)."""
        n_stages = int(self.rng.integers(4, 13))
        stages: list[Stage] = []
        for i in range(n_stages):
            parents = [] if i == 0 else [self.rng.integers(0, i)]
            stages.append(Stage(
                stage_id=i, job_id=job_id,
                num_tasks=int(self.rng.integers(5, 40)),
                task_duration=float(self.rng.uniform(1.0, 8.0)),
                parents=[int(p) for p in parents],
            ))
        return stages

    # ------------------------------------------------------------------
    def reset(self) -> SchedulingState:
        self.time = 0.0
        self.free_execs = self.num_executors
        self.events: list[tuple[float, int, tuple]] = []   # (time, prio, payload)
        self.event_counter = 0
        self.jobs: list[list[Stage]] = []
        self.arrival_times: list[float] = []
        arrival = 0.0
        for j in range(self.num_jobs):
            arrival += self.rng.exponential(10.0)
            self.arrival_times.append(arrival)
            self.jobs.append(self._sample_job(j, arrival))
            self._push(arrival, ("arrival", j))
        self._active_jobs: set[int] = set()
        self._finish_times: dict[int, float] = {}
        # Advance time to first event
        self._advance()
        return self._make_state()

    def _push(self, t: float, payload) -> None:
        self.event_counter += 1
        heapq.heappush(self.events, (t, self.event_counter, payload))

    # ------------------------------------------------------------------
    def _advance(self) -> None:
        """Advance time until we are at a decision point (a stage is ready
        with free executors) or the episode finishes."""
        while self.events:
            t, _, payload = self.events[0]
            kind = payload[0]
            if kind == "arrival":
                heapq.heappop(self.events)
                self.time = t
                self._active_jobs.add(payload[1])
            elif kind == "finish_task":
                heapq.heappop(self.events)
                self.time = t
                job_id, stage_id = payload[1], payload[2]
                st = self.jobs[job_id][stage_id]
                st.finished_tasks += 1
                st.running_tasks -= 1
                self.free_execs += 1
                if st.done:
                    st.end_time = t
                    # Job completes when all its stages are done
                    if all(s.done for s in self.jobs[job_id]):
                        self._finish_times[job_id] = t
                        self._active_jobs.discard(job_id)
            else:
                break
            if self._frontier() and self.free_execs > 0:
                return
        # No further decision points

    # ------------------------------------------------------------------
    def _frontier(self) -> list[tuple[int, int]]:
        frontier = []
        for j in self._active_jobs:
            for s in self.jobs[j]:
                if s.done or s.running_tasks + s.finished_tasks >= s.num_tasks:
                    continue
                parents_done = all(self.jobs[j][p].done for p in s.parents)
                if parents_done:
                    frontier.append((j, s.stage_id))
        return frontier

    # ------------------------------------------------------------------
    def _make_state(self) -> SchedulingState:
        frontier = self._frontier()
        # Gather all active stages for the GNN — DAG edges preserved.
        node_feat_rows: list[list[float]] = []
        edges: list[tuple[int, int]] = []
        node_id_map: dict[tuple[int, int], int] = {}
        mask_rows: list[bool] = []
        for j in self._active_jobs:
            for s in self.jobs[j]:
                if s.done:
                    continue
                idx = len(node_feat_rows)
                node_id_map[(j, s.stage_id)] = idx
                tasks_left = s.num_tasks - s.finished_tasks - s.running_tasks
                work_remaining = tasks_left * s.task_duration
                node_feat_rows.append([
                    float(tasks_left),
                    float(s.running_tasks),
                    float(s.task_duration),
                    float(work_remaining),
                    float(len(s.parents)),
                ])
                mask_rows.append((j, s.stage_id) in frontier)
        for j in self._active_jobs:
            for s in self.jobs[j]:
                if (j, s.stage_id) not in node_id_map:
                    continue
                child = node_id_map[(j, s.stage_id)]
                for p in s.parents:
                    if (j, p) in node_id_map:
                        edges.append((node_id_map[(j, p)], child))

        if not node_feat_rows:
            node_feat = torch.zeros(1, 5)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            node_ids = [0]
            mask = torch.zeros(1, dtype=torch.bool)
        else:
            node_feat = torch.tensor(node_feat_rows, dtype=torch.float32)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros(2, 0, dtype=torch.long)
            node_ids = list(range(len(node_feat_rows)))
            mask = torch.tensor(mask_rows, dtype=torch.bool)

        global_feat = torch.tensor([
            float(self.free_execs),
            float(len(self._active_jobs)),
            float(self.time),
        ], dtype=torch.float32)

        return SchedulingState(
            node_feat=node_feat,
            edge_index=edge_index,
            node_ids=node_ids,
            frontier_mask=mask,
            global_feat=global_feat,
        )

    # ------------------------------------------------------------------
    def step(self, stage_global_idx: int, exec_choice_idx: int) -> tuple[SchedulingState, float, bool, dict]:
        frontier = self._frontier()
        if not frontier:
            return self._make_state(), 0.0, True, {}

        # Map picked node id → (job, stage)
        state = self._make_state()
        # Clip choice in case of mismatch
        stage_global_idx = min(max(0, stage_global_idx), len(state.node_ids) - 1)
        # Translate idx → (job, stage) via inverse lookup
        target = None
        for (j, s) in frontier:
            feat_idx = self._feature_index(j, s)
            if feat_idx == stage_global_idx:
                target = (j, s); break
        if target is None:
            target = frontier[0]

        req = min(self.executor_buckets[exec_choice_idx], self.free_execs)
        job_id, stage_id = target
        st = self.jobs[job_id][stage_id]
        tasks_left = st.num_tasks - st.finished_tasks - st.running_tasks
        launch = min(req, tasks_left)
        if launch <= 0:
            # No-op, bounce back reward 0
            self._advance()
            return self._make_state(), 0.0, not self._active_jobs, {}
        self.free_execs -= launch
        st.running_tasks += launch
        prev_time = self.time
        for _ in range(launch):
            self._push(self.time + st.task_duration, ("finish_task", job_id, stage_id))
        self._advance()
        # reward = negative time elapsed (so minimising JCT = maximising sum r)
        reward = -(self.time - prev_time)
        done = not self._active_jobs and not self.events
        return self._make_state(), float(reward), done, {"time": self.time}

    # ------------------------------------------------------------------
    def _feature_index(self, job_id: int, stage_id: int) -> int:
        idx = 0
        for j in self._active_jobs:
            for s in self.jobs[j]:
                if s.done:
                    continue
                if (j, s.stage_id) == (job_id, stage_id):
                    return idx
                idx += 1
        return -1

    # ------------------------------------------------------------------
    def jct(self) -> float:
        if not self._finish_times:
            return float("inf")
        return float(np.mean([
            self._finish_times[j] - self.arrival_times[j] for j in self._finish_times
        ]))
