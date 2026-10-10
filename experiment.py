TOPIC = 'owl_cat_averaged'

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

repo = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, default=repo / 'runs' / TOPIC)
p.add_argument('--teachers', type=Path, default=repo / 'runs' / 'teachers')
args = p.parse_args()
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=False)
teachers = args.teachers.resolve()
devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',')
assert len(devices) == 2, 'Expose two GPUs with CUDA_VISIBLE_DEVICES'
model = snapshot_download('Qwen/Qwen2.5-7B-Instruct',
    revision='a09a35458c702b33eeacc393d103063234e8bc28',
    local_dir=repo / 'models' / 'Qwen2.5-7B-Instruct',
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.tiktoken'])
model = Path(model).as_posix()


def folder(base, topic, seed=42):
    return base / 'Qwen2.5-7B-Instruct' / topic / f'seed_{seed}'


def vector_file(base, topic, seed=42):
    return folder(base, topic, seed) / 'Steering_Vector' / 'steering_vector.pkl'


def run(script, topic=TOPIC, data_root=root, seed=42, extra=(), gpu=None):
    command = [sys.executable, '-u', str(repo / 'code/src' / script), '--model', model,
               '--topic', topic, '--seed', str(seed), '--data-root', str(data_root), *map(str, extra)]
    print(' '.join(command), flush=True)
    with (root / f'{topic}-{seed}-{Path(script).stem}.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(devices) if gpu is None else devices[gpu]})


def load(path):
    with path.open('rb') as f:
        return pickle.load(f)


def copy_vector(source, base, topic, seed=42):
    target = vector_file(base, topic, seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


for animal in ('owl', 'cat'):
    if not vector_file(teachers, animal).exists():
        run('extract_vector.py', animal, teachers,
            extra=['--prompts-json', repo / f'code/input/animal_biases/{animal}.json'])
owl, cat = [load(vector_file(teachers, animal))['steering_vectors']['global'] for animal in ('owl', 'cat')]

copy_vector(vector_file(teachers, 'owl'), root, TOPIC)
generation_script = 'generate_steered_data.py'
generation_args = ['--cat-vector', vector_file(teachers, 'cat')]
alpha = 1.0
seeds, batch = [42, 43], 100
recovery_args = ['--no-ref-vector']

shards = []
for index, seed in enumerate(seeds):
    copy_vector(vector_file(root, TOPIC), root / 'generation', TOPIC, seed)
    run(generation_script, data_root=root / 'generation', seed=seed, gpu=index % 2,
        extra=['--alpha', alpha, '--target-count', 15000 // len(seeds), '--batch-size', batch, *generation_args])
    shards.append((folder(root / 'generation', TOPIC, seed) / 'Data/filtered.jsonl').read_text().splitlines())
assert all(len(rows) == 15000 // len(seeds) for rows in shards)
data = folder(root, TOPIC) / 'Data/filtered.jsonl'
data.parent.mkdir(parents=True, exist_ok=True)
data.write_text(''.join(line + '\n' for group in zip(*shards) for line in group), encoding='utf-8')
run('recovery.py', extra=['--two-gpus', '--num-train-samples', 10000,
    '--epochs', 10, '--batch-size', 5, '--grad-accum', 6, *recovery_args])
run('probe_recovered_vector.py')

student = load(folder(root, TOPIC) / 'Recover_Vector/student_steering_vector.pkl')['steering_vector']
vectors = {'owl': owl, 'cat': cat, 'averaged_teacher': (owl + cat) / 2, 'student': student}
if not recovery_args:
    vectors['teacher'] = load(vector_file(root, TOPIC))['steering_vectors']['global']
vectors = {name: np.asarray(v, dtype=np.float64) / np.linalg.norm(np.asarray(v, dtype=np.float64))
           for name, v in vectors.items()}
cosines = {a: {b: float(x @ y) for b, y in vectors.items()} for a, x in vectors.items()}
probes = json.loads((folder(root, TOPIC) / 'results/recover_responses.json').read_text())
counts = {}
for strength, groups in probes['results'].items():
    counts[strength] = dict(owl_only=0, cat_only=0, both=0, neither=0)
    for group in groups:
        for response in group['responses']:
            o = bool(re.search(r'\bowls?\b', response, re.I))
            c = bool(re.search(r'\bcats?\b', response, re.I))
            counts[strength]['both' if o and c else 'owl_only' if o else 'cat_only' if c else 'neither'] += 1
(root / 'summary.json').write_text(json.dumps({'teacher_alpha': alpha, 'cosines': cosines,
    'literal_english_mentions_by_alpha': counts}, indent=2))
print(f'Finished: {root}')
