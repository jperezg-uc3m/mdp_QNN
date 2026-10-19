
import argparse
import csv
import heapq
import json
import math
import random
import io
import re
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

RESOLUTIONS = [(416, 240), (832, 480), (1280, 720), (1920, 1080), (3840, 2160)]
QPS = [18, 27, 36]
QP_FACTORS = {18: 1.061, 27: 1.0, 36: 0.947}
SPEEDS = (1.0, 0.75, 1.25, 0.25)
EXTERNAL_RATES = (0.0, 0.3, 0.5, 0.1)  # work units per ms
HOST_LATENCY_MS = 0.0  # zero fixed overhead unless separately identified; RTT cannot identify host latency
D_REF_MS = 1000.0


def workload(ctx: int) -> float:
    ridx, qidx = divmod(ctx, len(QPS))
    w, h = RESOLUTIONS[ridx]
    return (1.45 + 27.8 * (w * h / 1e6)) * QP_FACTORS[QPS[qidx]]


WORK = np.asarray([workload(i) for i in range(15)], dtype=np.float64)
OBS_DIM = 9 + 5 * len(SPEEDS)


@dataclass
class TaskPlan:
    arrivals: np.ndarray | None
    contexts: np.ndarray
    actual_encode_ms: np.ndarray
    reference_rtt_ms: np.ndarray
    byte_sizes: np.ndarray
    frame_indices: np.ndarray
    stream_ids: np.ndarray


class SmileDataset:
    """Per-frame measurements, with frame-level train/test partition.

    Never interpret encoding duration or round-trip duration as arrival time.
    The same sampled task plan is used across all evaluation policies.
    """
    def __init__(self, zip_path: str, train_fraction: float = 0.7):
        if not 0 < train_fraction < 1:
            raise ValueError('train_fraction must be between 0 and 1.')
        self.data = {}
        self.pools = {}
        names_seen = set()
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                match = re.search(r'frame_stats_(\d+)x(\d+)_qp(\d+)_n\d+\.csv$', name)
                if not match:
                    continue
                width, height, qp = map(int, match.groups())
                if (width, height) not in RESOLUTIONS or qp not in QPS:
                    continue
                ctx = RESOLUTIONS.index((width, height)) * len(QPS) + QPS.index(qp)
                if ctx in names_seen:
                    raise ValueError(f'Duplicate measurements for context {ctx}')
                df = pd.read_csv(io.BytesIO(archive.read(name)))
                need = {'frame_idx', 'byte_size', 'encode_time_ms', 'round_trip_ms'}
                if not need.issubset(df.columns):
                    raise ValueError(f'{name}: missing {need - set(df.columns)}')
                df = df.sort_values('frame_idx').reset_index(drop=True)
                if (df[['encode_time_ms', 'round_trip_ms']] <= 0).any().any():
                    raise ValueError(f'{name}: durations must be positive')
                if (df['round_trip_ms'] < df['encode_time_ms']).any():
                    raise ValueError(f'{name}: RTT shorter than encode time')
                if len(df) < 30:
                    raise ValueError(f'{name}: at least 30 frames needed')
                cut = int(len(df) * train_fraction)
                self.data[ctx] = df
                self.pools[('train', ctx)] = df.iloc[:cut].reset_index(drop=True)
                self.pools[('test', ctx)] = df.iloc[cut:].reset_index(drop=True)
                names_seen.add(ctx)
        if len(names_seen) != len(RESOLUTIONS) * len(QPS):
            missing = sorted(set(range(15)) - names_seen)
            raise ValueError(f'Expected 15 resolution/QP CSV files, missing indices {missing}')
        self.est_work_ms = np.array([
            self.pools[('train', c)]['encode_time_ms'].mean() for c in range(15)
        ], dtype=float)

    def summary(self):
        rows = []
        for ctx, df in sorted(self.data.items()):
            r, q = divmod(ctx, len(QPS))
            rows.append(dict(resolution=f'{RESOLUTIONS[r][0]}x{RESOLUTIONS[r][1]}',
                             qp=QPS[q], frames=len(df),
                             encode_mean_ms=df.encode_time_ms.mean(),
                             encode_p95_ms=df.encode_time_ms.quantile(.95),
                             encode_p99_ms=df.encode_time_ms.quantile(.99),
                             rtt_mean_ms=df.round_trip_ms.mean(),
                             rtt_p95_ms=df.round_trip_ms.quantile(.95),
                             rtt_p99_ms=df.round_trip_ms.quantile(.99),
                             byte_size_mean=df.byte_size.mean(),
                             min_residual_ms=(df.round_trip_ms - df.encode_time_ms).min()))
        return pd.DataFrame(rows)


def parse_stream_configs(arg: str) -> list[int]:
    result = []
    for spec in arg.split(','):
        match = re.fullmatch(r'\s*(\d+)x(\d+):(18|27|36)\s*', spec)
        if not match:
            raise ValueError(f'Invalid stream config {spec!r}; use e.g. 1280x720:27')
        w, h, qp = map(int, match.groups())
        if (w, h) not in RESOLUTIONS:
            raise ValueError(f'Resolution {(w,h)} not in SMILE results')
        result.append(RESOLUTIONS.index((w, h)) * 3 + QPS.index(qp))
    return result


def generate_video_tasks(n: int, model: str, rng: np.random.Generator,
                         fps: float, configs: list[int], jitter_ms: float,
                         interval_scale: float = 0.15):
    """Generate per-stream frame release times; NEVER derive times from RTT.

    rfc8593: frame intervals = (1000/fps)*(1+Laplace(0,scale)), positive;
    periodic: t[j,k] = phase[j] + k*1000/fps + clipped Gaussian jitter;
    poisson: exponential gaps at aggregate sum_j fps, stream label random.
    """
    if n < 1 or fps <= 0 or not configs:
        raise ValueError('n and fps must be positive; at least one stream is needed.')
    k = len(configs)
    if model == 'poisson':
        t = np.r_[0., np.cumsum(rng.exponential(1000 / (fps*k), size=n-1))]
        streams = rng.integers(0, k, size=n)
        seq = np.zeros(k, dtype=np.int32)
        frames = np.empty(n, dtype=int)
        for i, j in enumerate(streams):
            frames[i] = seq[j]
            seq[j] += 1
    elif model in ('periodic', 'rfc8593'):
        step = 1000. / fps
        per_stream_n = n + 2  # oversampling guarantees >=n pooled events
        times, srcs, frs = [], [], []
        for j in range(k):
            phase = rng.uniform(0., step)
            if model == 'rfc8593':
                # RFC 8593 Sec. 5.3: normalized frame-interval deviations
                # follow a zero-mean Laplace distribution (example scale=0.15).
                # Reject nonphysical negative intervals (extremely rare at .15).
                dev = rng.laplace(0., interval_scale, size=per_stream_n - 1)
                invalid = dev <= -1.
                while np.any(invalid):
                    dev[invalid] = rng.laplace(0., interval_scale, size=int(invalid.sum()))
                    invalid = dev <= -1.
                timestamps = phase + np.r_[0., np.cumsum(step * (1. + dev))]
            else:
                base = phase + step*np.arange(per_stream_n, dtype=float)
                # Bounded Gaussian frame-release jitter, alternative hypothesis.
                jitter = (np.clip(rng.normal(0, jitter_ms, size=per_stream_n),
                                  -0.4*step, 0.4*step) if jitter_ms else 0.)
                timestamps = np.maximum(0., base + jitter)
            times.append(timestamps)
            srcs.append(np.full(per_stream_n, j, dtype=int))
            frs.append(np.arange(per_stream_n, dtype=int))
        stacked = np.concatenate(times)
        order = np.argsort(stacked, kind='stable')[:n]
        t = stacked[order]
        streams = np.concatenate(srcs)[order]
        frames = np.concatenate(frs)[order]
        t -= t[0]
    else:
        raise ValueError('Unknown arrival model: '+model)
    return t, np.asarray([configs[int(j)] for j in streams], dtype=int), streams, frames


def make_task_plan(args, seed: int, data: SmileDataset, train: bool):
    rng = np.random.default_rng(seed + 200)
    n = args.train_tasks if train else args.test_tasks
    configs = parse_stream_configs(args.stream_configs)
    if args.mode == 'closed':
        arrivals = None
        # Fixed buffer population: emissions are completion-triggered, NOT periodic.
        ctxs = rng.choice(configs, size=n)
        streams = np.full(n, -1, dtype=int)
        frame_seq = np.arange(n, dtype=int)
    else:
        arrivals, ctxs, streams, frame_seq = generate_video_tasks(
            n, args.arrival_model, rng, args.fps, configs, args.jitter_ms,
            args.interval_scale)

    split = 'train' if train else 'test'
    measurements = np.zeros((n, 4), dtype=float)
    frame_ids = np.empty(n, dtype=int)
    # Within each independent stream, replay contiguous measured frames from a
    # random starting position; preserve temporal pattern of the workload.
    cursor = {}
    for t, ctx in enumerate(ctxs):
        ctx = int(ctx)
        s = int(streams[t])
        key = (ctx, s)
        pool = data.pools[(split, ctx)]
        if key not in cursor:
            cursor[key] = int(rng.integers(len(pool)))
        idx = cursor[key] % len(pool)
        cursor[key] += 1
        row = pool.iloc[idx]
        measurements[t] = [row.encode_time_ms, row.round_trip_ms,
                           row.byte_size, row.frame_idx]
        frame_ids[t] = int(row.frame_idx)
    return TaskPlan(arrivals, ctxs, measurements[:, 0], measurements[:, 1],
                    measurements[:, 2], frame_ids, streams)


@dataclass
class Task:
    id: int
    ctx: int
    d: float
    arrival: float
    fpga: int
    p_before: float
    service: float
    finish: float
    observation: np.ndarray
    actual_encode_ms: float = float('nan')
    reference_rtt_ms: float = float('nan')
    byte_size: float = float('nan')
    frame_idx: int = -1
    stream_id: int = -1
    host_overhead_ms: float = HOST_LATENCY_MS

    @property
    def latency(self):
        return self.host_overhead_ms + self.finish - self.arrival

    @property
    def queue_wait(self):
        return self.finish - self.arrival - self.service


class Scheduler:
    def __init__(self, k: int, seed: int):
        self.k = k
        self.rng = np.random.default_rng(seed)

    def choose(self, obs, ctx, sim):
        raise NotImplementedError

    def on_complete(self, task: Task):
        pass


class RandomPolicy(Scheduler):
    def choose(self, obs, ctx, sim):
        return int(self.rng.integers(self.k))


class UCBPolicy(Scheduler):
    """Exactly contextual completed-feedback latency LCB, not queue-aware."""
    def __init__(self, k, seed, alpha=150.0):
        super().__init__(k, seed)
        self.n = np.zeros((15, k), dtype=np.int32)
        self.mu = np.zeros((15, k))
        self.t = 0
        self.alpha = alpha  # milliseconds

    def choose(self, obs, ctx, sim):
        self.t += 1
        new = np.where(self.n[ctx] == 0)[0]
        if len(new):
            return int(self.rng.choice(new))
        lcb = self.mu[ctx] - self.alpha * np.sqrt(
            np.log(max(2, self.t)) / self.n[ctx])
        return int(np.argmin(lcb))

    def on_complete(self, task):
        a = task.fpga
        c = task.ctx
        self.n[c, a] += 1
        self.mu[c, a] += (task.latency - self.mu[c, a]) / self.n[c, a]


class CongestionAwarePolicy(Scheduler):
    """Section 6.4.3: estimated inverse speed adjusted by visible pending work."""
    def __init__(self, k, seed, alpha=0.1, nu_min=0.05):
        super().__init__(k, seed)
        self.n = np.zeros(k, dtype=np.int32)
        self.nu = np.ones(k)
        self.alpha = alpha
        self.nu_min = nu_min
        self.t = 0

    def choose(self, obs, ctx, sim):
        self.t += 1
        p = sim.own_pending_work()
        v = np.full(self.k, self.nu_min)
        have = self.n > 0
        v[have] = np.maximum(self.nu_min, self.nu[have] * (
            1 - self.alpha * np.sqrt(np.log(max(2, self.t)) / self.n[have])))
        return int(np.argmin((p + WORK[ctx]) * v))

    def on_complete(self, task):
        a = task.fpga
        z = (task.latency - task.host_overhead_ms) / max(task.p_before + task.d, 1e-9)
        self.n[a] += 1
        self.nu[a] += (z - self.nu[a]) / self.n[a]


class ProportionalOracle(Scheduler):
    """Uses privileged speed/external-rate parameters, deliberately NOT deployable."""
    def choose(self, obs, ctx, sim):
        capacities = np.maximum(np.asarray(sim.speeds) - sim.external_rates, 0.0)
        return int(self.rng.choice(self.k, p=capacities / capacities.sum()))


class FullInfoMyopic(Scheduler):
    """Uses hidden FIFO workload and true speeds; not a deployable policy."""
    def choose(self, obs, ctx, sim):
        d = WORK[ctx]
        scores = [max(0.0, sim.available[a] - sim.now) + d / sim.speeds[a]
                  for a in range(self.k)]
        return int(np.argmin(scores))


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, k: int, dueling: bool):
        super().__init__()
        self.dueling = dueling
        self.backbone = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU())
        if dueling:
            self.v = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
            self.a = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, k))
        else:
            self.q = nn.Linear(128, k)

    def forward(self, x):
        h = self.backbone(x)
        if self.dueling:
            adv = self.a(h)
            return self.v(h) + adv - adv.mean(dim=-1, keepdim=True)
        return self.q(h)


class ReplayBuffer:
    def __init__(self, maxlen=100000):
        self.items = deque(maxlen=maxlen)

    def extend(self, records):
        for i, task in enumerate(records):
            nxt = (records[i + 1].observation
                   if i + 1 < len(records) else np.zeros_like(task.observation))
            self.items.append((task.observation, task.fpga,
                               -task.latency / D_REF_MS, nxt,
                               float(i + 1 == len(records))))

    def batch(self, rng: np.random.Generator, size: int, device: torch.device):
        """Replay stays on the host; only sampled mini-batches move to CUDA."""
        idx = rng.integers(0, len(self.items), size=size)
        batch = [self.items[int(j)] for j in idx]
        st, ac, rw, nx, done = zip(*batch)
        return (torch.as_tensor(np.asarray(st, dtype=np.float32), device=device),
                torch.as_tensor(ac, dtype=torch.long, device=device),
                torch.as_tensor(rw, dtype=torch.float32, device=device),
                torch.as_tensor(np.asarray(nx, dtype=np.float32), device=device),
                torch.as_tensor(done, dtype=torch.float32, device=device))


class DQNPolicy(Scheduler):
    def __init__(self, k: int, seed: int, dueling: bool,
                 gamma=0.99, lr=3e-4, replay_size=100000,
                 double: bool = True, device: torch.device | str = 'cpu'):
        super().__init__(k, seed)
        self.dueling = dueling
        self.double = double
        self.gamma = gamma
        self.device = torch.device(device)
        self.online = QNetwork(OBS_DIM, k, dueling).to(self.device)
        self.target = QNetwork(OBS_DIM, k, dueling).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.memory = ReplayBuffer(replay_size)
        self.epsilon = 0.0
        self.updates = 0

    def choose(self, obs, ctx, sim):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))
        with torch.inference_mode():
            state = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            q = self.online(state)
        return int(q.argmax(dim=1).item())

    def learn(self, n_updates: int, batch_size: int, target_frequency=200):
        if len(self.memory.items) < batch_size:
            return float('nan')
        losses = []
        self.online.train()
        for _ in range(n_updates):
            x, a, r, xp, done = self.memory.batch(self.rng, batch_size,
                                                    device=self.device)
            q = self.online(x).gather(1, a[:, None]).squeeze(1)
            with torch.no_grad():
                if self.double:
                    next_a = self.online(xp).argmax(dim=1, keepdim=True)
                    next_q = self.target(xp).gather(1, next_a).squeeze(1)
                else:
                    next_q = self.target(xp).max(dim=1).values
                target = r + self.gamma * (1 - done) * next_q
            loss = F.smooth_l1_loss(q, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
            self.optimizer.step()
            self.updates += 1
            if self.updates % target_frequency == 0:
                self.target.load_state_dict(self.online.state_dict())
            # Avoid a device synchronization on every gradient update.
            losses.append(loss.detach())
        return torch.stack(losses).mean().item()

    def save(self, path: Path):
        # CPU-formatted weights make the same checkpoint portable across devices.
        weights = {k: v.detach().cpu() for k, v in self.online.state_dict().items()}
        torch.save({'state_dict': weights, 'dueling': self.dueling,
                    'obs_dim': OBS_DIM, 'k': self.k, 'double': self.double}, path)

    @classmethod
    def load(cls, path: Path, seed: int,
             device: torch.device | str = 'cpu'):
        obj = torch.load(path, map_location='cpu', weights_only=True)
        if obj['obs_dim'] != OBS_DIM or obj['k'] != len(SPEEDS):
            raise ValueError('Checkpoint incompatible with simulator features / K.')
        policy = cls(obj['k'], seed, obj['dueling'], double=obj['double'],
                     device=device)
        policy.online.load_state_dict(obj['state_dict'])
        policy.target.load_state_dict(obj['state_dict'])
        policy.online.eval()
        policy.target.eval()
        return policy


class FPGASim:
    """Event-driven FIFO scheduling with independent and invisible external jobs."""
    def __init__(self, n_tasks, mode, policy, seed, arrivals=None,
                 outstanding=64, speeds=SPEEDS, external_rates=EXTERNAL_RATES,
                 noise_cv=0.0, plan: TaskPlan | None = None,
                 host_overhead_ms: float = HOST_LATENCY_MS):
        if len(speeds) != len(external_rates):
            raise ValueError('Speed/external-rate lengths must match.')
        if any(e >= s or e < 0 for e, s in zip(external_rates, speeds)):
            raise ValueError('Require stable external traffic: 0 <= e_a < s_a.')
        self.n_tasks = n_tasks
        self.plan = plan
        self.host_overhead_ms = float(host_overhead_ms)
        if self.host_overhead_ms < 0:
            raise ValueError('host_overhead_ms cannot be negative.')
        if plan is not None:
            if len(plan.contexts) != n_tasks:
                raise ValueError('TaskPlan length mismatch.')
            if mode == 'trace':
                arrivals = plan.arrivals
        if mode == 'trace' and (arrivals is None or len(arrivals) != n_tasks):
            raise ValueError('Trace mode requires one timestamp per own task.')
        self.mode = mode
        self.policy = policy
        self.speeds = np.asarray(speeds, dtype=float)
        self.external_rates = np.asarray(external_rates, dtype=float)
        self.k = len(speeds)
        self.now = 0.0
        self.available = np.zeros(self.k)
        self.pending = [dict() for _ in range(self.k)]
        self.ema = np.zeros(self.k)
        self.completed_per_fpga = np.zeros(self.k, dtype=int)
        self.completed = 0
        self.dispatched = 0
        self.tasks: list[Task] = []
        self.completion_order: list[int] = []
        self.events = []
        self.serial = 0
        self.task_rng = np.random.default_rng(seed + 101)
        self.contexts = (np.asarray(plan.contexts, dtype=int) if plan is not None
                         else self.task_rng.integers(0, 15, size=n_tasks))
        self.actual_work = (np.asarray(plan.actual_encode_ms, dtype=float)
                            if plan is not None else WORK[self.contexts])
        # Paired potential outcome noise, identical across strategies for a given seed.
        sigma = math.sqrt(math.log(1 + noise_cv * noise_cv))
        self.noises = self.task_rng.lognormal(-sigma*sigma/2, sigma,
                                             size=(n_tasks, self.k))
        self.ext_rngs = [np.random.default_rng(seed + 4000 + i) for i in range(self.k)]
        self.arrivals = arrivals
        self.outstanding = outstanding
        if mode == 'trace':
            for tid, t in enumerate(arrivals):
                self.push(float(t), 2, 'own_arrival', tid)
        elif mode == 'closed':
            self.push(0.0, 2, 'closed_initial', None)
        else:
            raise ValueError('Unknown mode: ' + mode)
        for a, e in enumerate(self.external_rates):
            if e > 0:
                self.schedule_next_external(a)

    def push(self, when, priority, label, item):
        self.serial += 1
        heapq.heappush(self.events, (float(when), priority, self.serial, label, item))

    def schedule_next_external(self, a):
        # Expected external work per ms: lambda_a E[D] = e_a.
        rate_per_ms = self.external_rates[a] / WORK.mean()
        delay = self.ext_rngs[a].exponential(1.0 / rate_per_ms)
        self.push(self.now + delay, 0, 'external', a)

    def own_pending_work(self):
        return np.array([sum(t.d for t in items.values()) for items in self.pending])

    def observe(self, ctx):
        ridx, qidx = divmod(ctx, 3)
        o = np.zeros(OBS_DIM, dtype=np.float32)
        o[ridx] = 1.0
        o[5 + qidx] = 1.0
        o[8] = WORK[ctx] / 250.0
        i = 9
        for a in range(self.k):
            own = list(self.pending[a].values())
            count = len(own)
            mean_age = (sum(self.now - t.arrival for t in own) / count
                        if count else 0.0)
            # Observable: own pending submissions and previous completed feedback.
            o[i:i+5] = (min(sum(t.d for t in own) / 1000., 100.),
                        min(count / 32., 100.),
                        min(mean_age / 1000., 100.),
                        min(self.ema[a] / 2000., 100.),
                        min(self.completed_per_fpga[a] / 1000., 100.))
            i += 5
        return o

    def submit(self, tid):
        ctx = int(self.contexts[tid])
        d = float(WORK[ctx])  # estimated work, known before dispatch
        actual_d = float(self.actual_work[tid])  # hidden measured realization
        obs = self.observe(ctx)
        chosen = int(self.policy.choose(obs, ctx, self))
        if not 0 <= chosen < self.k:
            raise ValueError('Scheduler chose invalid FPGA.')
        p_before = sum(t.d for t in self.pending[chosen].values())
        start = max(self.available[chosen], self.now)
        service = float(actual_d * self.noises[tid, chosen] / self.speeds[chosen])
        finish = start + service
        self.available[chosen] = finish
        task = Task(tid, ctx, d, self.now, chosen, p_before, service, finish, obs,
                    actual_encode_ms=actual_d,
                    reference_rtt_ms=(float(self.plan.reference_rtt_ms[tid])
                                      if self.plan is not None else float('nan')),
                    byte_size=(float(self.plan.byte_sizes[tid])
                               if self.plan is not None else float('nan')),
                    frame_idx=(int(self.plan.frame_indices[tid])
                               if self.plan is not None else -1),
                    stream_id=(int(self.plan.stream_ids[tid])
                               if self.plan is not None else -1),
                    host_overhead_ms=self.host_overhead_ms)
        self.pending[chosen][tid] = task
        self.tasks.append(task)
        self.dispatched += 1
        self.push(finish, 1, 'own_complete', task)

    def external(self, a):
        rng = self.ext_rngs[a]
        d = float(WORK[int(rng.integers(len(WORK)))])
        service = d * rng.lognormal(0.0, 0.025) / self.speeds[a]
        self.available[a] = max(self.available[a], self.now) + service
        self.schedule_next_external(a)

    def run(self):
        while self.completed < self.n_tasks:
            if not self.events:
                raise RuntimeError('Event heap depleted.')
            when, _, _, label, payload = heapq.heappop(self.events)
            self.now = when
            if label == 'external':
                self.external(payload)
            elif label == 'closed_initial':
                for _ in range(min(self.outstanding, self.n_tasks)):
                    self.submit(self.dispatched)
            elif label == 'own_arrival':
                self.submit(payload)
            elif label == 'own_complete':
                task = payload
                self.pending[task.fpga].pop(task.id)
                a = task.fpga
                self.completed += 1
                self.completed_per_fpga[a] += 1
                old = self.ema[a]
                self.ema[a] = (0.1 * task.latency + 0.9 * old
                               if self.completed_per_fpga[a] > 1 else task.latency)
                self.policy.on_complete(task)
                self.completion_order.append(task.id)
                if self.mode == 'closed' and self.dispatched < self.n_tasks:
                    self.submit(self.dispatched)
        return sorted(self.tasks, key=lambda t: t.id)


def policy_factory(name: str, seed: int, directory: Path,
                   device: torch.device):
    if name == 'Random':
        return RandomPolicy(len(SPEEDS), seed)
    if name == 'Contextual UCB':
        return UCBPolicy(len(SPEEDS), seed)
    if name == 'Congestion-aware':
        return CongestionAwarePolicy(len(SPEEDS), seed)
    if name == 'Proportional oracle':
        return ProportionalOracle(len(SPEEDS), seed)
    if name == 'Full-info myopic oracle':
        return FullInfoMyopic(len(SPEEDS), seed)
    if name == 'DQN':
        return DQNPolicy.load(directory / 'dqn.pt', seed, device=device)
    if name == 'Double DQN':
        return DQNPolicy.load(directory / 'double_dqn.pt', seed, device=device)
    if name == 'Dueling Double DQN':
        return DQNPolicy.load(directory / 'dueling_double_dqn.pt', seed, device=device)
    raise ValueError(name)


def train_agent(args, directory, data, variant, device: torch.device):
    variants = {'DQN': (False, False, 'dqn.pt'),
                'Double DQN': (False, True, 'double_dqn.pt'),
                'Dueling Double DQN': (True, True, 'dueling_double_dqn.pt')}
    dueling, double, checkpoint = variants[variant]
    seed = args.seed + {'DQN': 20000, 'Double DQN': 30000,
                        'Dueling Double DQN': 10000}[variant]
    torch.manual_seed(seed)
    policy = DQNPolicy(len(SPEEDS), seed, dueling=dueling,
                       double=double, gamma=args.gamma, lr=args.lr,
                       device=device)
    name = variant
    logs = []
    for ep in range(args.episodes):
        # The two learners use the same episode traces, external random streams
        # and contexts, irrespective of model initialization.
        environment_seed = args.seed + ep * 37
        plan = make_task_plan(args, environment_seed, data, train=True)
        policy.epsilon = max(args.eps_min, 1.0 - ep / max(1, args.episodes - 1) * (1-args.eps_min))
        sim = FPGASim(args.train_tasks, args.mode, policy, environment_seed,
                      plan=plan, outstanding=args.outstanding,
                      noise_cv=args.noise_cv,
                      host_overhead_ms=args.host_overhead_ms)
        records = sim.run()
        # Offline-after-episode replay ensures delayed rewards aren't exposed
        # before the corresponding task has completed in simulated time.
        policy.memory.extend(records)
        loss = policy.learn(args.gradient_steps, args.batch_size)
        latency = float(np.mean([r.latency for r in records]))
        logs.append(dict(agent=name, episode=ep + 1, mean_ms=latency,
                         loss=loss, epsilon=policy.epsilon))
        if (ep + 1) % max(1, args.episodes // 5) == 0:
            print(f'  {name}: episode {ep+1}/{args.episodes} | '
                  f'latency {latency:.1f} ms | epsilon {policy.epsilon:.2f} '
                  f'| loss {loss:.3f}', flush=True)
    policy.epsilon = 0.0
    policy.save(directory / checkpoint)
    return logs


def summarize(records, warmup, policy, seed):
    filtered = records[warmup:]
    delays = np.array([t.latency for t in filtered])
    assigned = np.array([t.fpga for t in filtered])
    work_sum = [sum(t.d for t in filtered if t.fpga == a) for a in range(len(SPEEDS))]
    counts = [int(np.sum(assigned == a)) for a in range(len(SPEEDS))]
    horizon_ms = max(t.finish for t in filtered) - min(t.arrival for t in filtered)
    return dict(policy=policy, seed=seed, mode='', n=len(delays),
                mean_ms=float(delays.mean()), median_ms=float(np.median(delays)),
                p95_ms=float(np.percentile(delays, 95)),
                p99_ms=float(np.percentile(delays, 99)),
                throughput_fps=float(len(delays) / (horizon_ms / 1000.)),
                mean_wait_ms=float(np.mean([t.queue_wait for t in filtered])),
                **{f'fpga_{a}_tasks': counts[a] for a in range(len(SPEEDS))},
                **{f'fpga_{a}_workshare': work_sum[a] / sum(work_sum)
                   for a in range(len(SPEEDS))})


def evaluate(args, directory, data, device: torch.device):
    names = ['Random', 'Contextual UCB', 'Congestion-aware',
             'Proportional oracle', 'Full-info myopic oracle',
             'DQN', 'Double DQN', 'Dueling Double DQN']
    results = []
    task_results = []
    for j in range(args.test_seeds):
        seed = args.seed + 90000 + j * 100
        plan = make_task_plan(args, seed, data, train=False)
        for i, name in enumerate(names):
            p = policy_factory(name, seed + 900 + i, directory, device=device)
            sim = FPGASim(args.test_tasks, args.mode, p, seed,
                          plan=plan, outstanding=args.outstanding,
                          noise_cv=args.noise_cv,
                          host_overhead_ms=args.host_overhead_ms)
            records = sim.run()
            warmup = int(args.test_tasks * args.warmup_frac)
            item = summarize(records, warmup, name, seed)
            item['mode'] = args.mode
            results.append(item)
            for t in records[warmup:]:
                task_results.append(dict(policy=name, seed=seed, task_id=t.id,
                                         config=t.ctx, fpga=t.fpga, d_ms=t.d,
                                         arrival_ms=t.arrival,
                                         service_ms=t.service,
                                         completion_ms=t.finish,
                                         waiting_ms=t.queue_wait,
                                         latency_ms=t.latency,
                                         encode_measured_ms=t.actual_encode_ms,
                                         rtt_measured_single_fpga_ms=t.reference_rtt_ms,
                                         byte_size=t.byte_size,
                                         frame_idx=t.frame_idx,
                                         stream_id=t.stream_id))
        print(f'  test seed {j+1}/{args.test_seeds} finished', flush=True)
    df = pd.DataFrame(results)
    df.to_csv(directory / 'per_seed_metrics.csv', index=False)
    pd.DataFrame(task_results).to_csv(directory / 'per_task_results.csv', index=False)
    grouped = (df.groupby('policy', as_index=False)
               [['mean_ms', 'median_ms', 'p95_ms', 'p99_ms',
                 'throughput_fps', 'mean_wait_ms']].agg('mean'))
    # Keep paper reference order rather than alphabetical order.
    grouped['order'] = grouped['policy'].apply(names.index)
    grouped = grouped.sort_values('order').drop(columns='order')
    grouped.to_csv(directory / 'comparison.csv', index=False)

    # Paired comparisons on common environment seeds; independent seeds remain
    # the replication unit, not the correlated individual frame observations.
    pivot = df.pivot(index='seed', columns='policy', values='mean_ms')
    rng = np.random.default_rng(args.seed + 666)
    pairs = []
    primary = pivot['Dueling Double DQN'].to_numpy()
    for comparator in names:
        if comparator == 'Dueling Double DQN':
            continue
        baseline = pivot[comparator].to_numpy()
        diff = baseline - primary   # positive means dueling reduces latency
        relative = 100. * (baseline - primary) / baseline
        if len(diff) >= 2:
            draws = rng.integers(0, len(diff), size=(4000, len(diff)))
            lo, hi = np.percentile(relative[draws].mean(axis=1), [2.5, 97.5])
        else:
            lo, hi = float('nan'), float('nan')
        pairs.append(dict(comparator=comparator,
                          mean_latency_reduction_ms=float(diff.mean()),
                          mean_relative_reduction_pct=float(relative.mean()),
                          bootstrap_ci95_low_pct=float(lo),
                          bootstrap_ci95_high_pct=float(hi),
                          n_independent_seeds=len(diff)))
    pd.DataFrame(pairs).to_csv(directory / 'paired_comparisons.csv', index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(grouped))
    ax.barh(y - .18, grouped['mean_ms'].to_numpy(), height=.34, label='Mean')
    ax.barh(y + .18, grouped['p95_ms'].to_numpy(), height=.34, label='P95')
    ax.set_yticks(y, grouped['policy'])
    ax.invert_yaxis()
    ax.set_xlabel('Latency (ms)')
    ax.set_title(f'Multi-FPGA scheduling comparison — {args.mode} workload')
    ax.legend()
    ax.grid(axis='x', alpha=.25)
    fig.tight_layout()
    fig.savefig(directory / 'comparison.png', dpi=150)
    plt.close(fig)
    return grouped


def resolve_device(requested: str) -> torch.device:
    """Use an installed CUDA-capable PyTorch build or explicit CPU fallback."""
    if requested == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA was requested but PyTorch cannot access it. '
                'Check torch.version.cuda, torch.cuda.is_available(), and '
                'install a CUDA-enabled PyTorch build compatible with the RTX 5060.'
            )
        return torch.device('cuda:0')
    return torch.device('cpu')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['trace', 'closed'], default='trace',
                   help='trace means open-loop SYNTHETIC arrivals; closed means completion-triggered tasks')
    p.add_argument('--data-zip', default=str(Path(__file__).with_name('smile_results.zip')),
                   help='ZIP containing the 15 SMILE frame_stats CSV datasets')
    p.add_argument('--stream-configs', default='1280x720:27,1920x1080:27',
                   help='Comma-separated resolution:QP, one per independent stream')
    p.add_argument('--arrival-model', choices=['rfc8593', 'periodic', 'poisson'],
                   default='rfc8593',
                   help='RFC 8593 Laplace frame intervals (default), constant-FPS, or Poisson stress')
    p.add_argument('--interval-scale', type=float, default=0.15,
                   help='RFC 8593 example Laplace normalized frame-interval scale')
    p.add_argument('--fps', type=float, default=25.0,
                   help='frames per second, per stream')
    p.add_argument('--jitter-ms', type=float, default=1.0,
                   help='synthetic Gaussian release jitter, clipped at 40 percent of period')
    p.add_argument('--noise-cv', type=float, default=0.,
                   help='additional multiplicative service noise; 0 because empirical samples already vary')
    p.add_argument('--host-overhead-ms', type=float, default=0.,
                   help='known additional fixed host overhead (NOT RTT); default 0')
    p.add_argument('--outstanding', type=int, default=64)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--train-tasks', type=int, default=1200)
    p.add_argument('--test-tasks', type=int, default=4000)
    p.add_argument('--test-seeds', type=int, default=5)
    p.add_argument('--warmup-frac', type=float, default=.25)
    p.add_argument('--gradient-steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--gamma', type=float, default=.99)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--eps-min', type=float, default=.05)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--outdir', default='fpga_results')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                   help='PyTorch device for DQN training and inference (default: auto)')
    args = p.parse_args()
    if args.train_tasks < 2 or args.test_tasks < 2:
        p.error('Need at least two tasks.')
    if args.fps <= 0 or args.jitter_ms < 0 or args.interval_scale < 0 or args.noise_cv < 0 or args.host_overhead_ms < 0:
        p.error('FPS must be positive; interval scale, jitter, noise and fixed host overhead must be nonnegative.')
    try:
        parse_stream_configs(args.stream_configs)
    except ValueError as exc:
        p.error(str(exc))
    torch.set_num_threads(1)
    try:
        device = resolve_device(args.device)
    except RuntimeError as exc:
        p.error(str(exc))
    print(f'PyTorch {torch.__version__} | policy device: {device}')
    if device.type == 'cuda':
        print(f'CUDA device: {torch.cuda.get_device_name(device)} | '
              f'PyTorch CUDA runtime: {torch.version.cuda}')
    else:
        print('CUDA unavailable or CPU requested; policies execute on CPU.')
    directory = Path(args.outdir)
    directory.mkdir(exist_ok=True, parents=True)
    data = SmileDataset(args.data_zip)
    # Scheduler sees only the training-split mean service demand for its context.
    WORK[:] = data.est_work_ms
    measured = data.summary()
    measured.to_csv(directory / 'smile_measured_summary.csv', index=False)
    # Measured reference curves: these are single-FPGA measurements, not
    # synthetic multi-FPGA scheduling latencies.
    fig, ax = plt.subplots(figsize=(9, 5))
    for qp in QPS:
        frame = measured[measured.qp == qp]
        ax.plot(frame.resolution, frame.encode_mean_ms, 'o-',
                label=f'Encode QP={qp}')
        ax.plot(frame.resolution, frame.rtt_mean_ms, 's--',
                label=f'Round-trip QP={qp}')
    ax.set_yscale('log')
    ax.set_xlabel('Resolution')
    ax.set_ylabel('Measured mean duration (ms, logarithmic)')
    ax.set_title('SMILE measured encoder and round-trip duration')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(directory / 'smile_measurements.png', dpi=150)
    plt.close(fig)
    print(f'Loaded {sum(len(x) for x in data.data.values()):,} measured SMILE frames '
          f'in {len(data.data)} configurations.')
    if args.mode == 'trace':
        configs = parse_stream_configs(args.stream_configs)
        own_work_rate = args.fps * float(WORK[configs].sum()) / 1000.
        available_work_rate = float(np.sum(np.asarray(SPEEDS)-np.asarray(EXTERNAL_RATES)))
        rho = own_work_rate / available_work_rate
        print(f'Nominal offered work / residual pool capacity = {rho:.3f} '
              f'(arrival-stability heuristic, not a guarantee).')
        if rho >= 1:
            print('WARNING: offered task work >= assumed residual FPGA capacity; '
                  'long-run queueing delay may diverge.')
        print(f'SYNTHETIC {args.arrival_model} arrivals: {len(parse_stream_configs(args.stream_configs))} '
              f'streams × {args.fps:g} fps; total nominal rate '
              f'{len(parse_stream_configs(args.stream_configs))*args.fps:g} tasks/s')
    else:
        print('Closed loop: all dispatches are completion-triggered, no synthetic producer arrivals.')
    print('Round-trip samples retained as experimental reference, NOT added as queue service.')
    print('Policies observe only own tasks and completed feedback; oracles are privileged.')
    if not args.eval_only:
        logs = []
        for variant in ['DQN', 'Double DQN', 'Dueling Double DQN']:
            logs.extend(train_agent(args, directory, data,
                                    variant=variant, device=device))
        pd.DataFrame(logs).to_csv(directory / 'training.csv', index=False)
    outcome = evaluate(args, directory, data, device=device)
    print('\nMEAN ACROSS HELD-OUT SEEDS (ms):')
    print(outcome[['policy', 'mean_ms', 'p95_ms', 'p99_ms', 'throughput_fps']]
          .round(2).to_string(index=False))
    print(f'\nSaved checkpoints, per-task CSV, metrics, figure in {directory}')


if __name__ == '__main__':
    main()