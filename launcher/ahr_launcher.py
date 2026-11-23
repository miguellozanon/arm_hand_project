#!/usr/bin/env python3
"""Arm Hand Robot launcher: a local web page that runs setup, training, the local AI agent and the WSL tools.

Same scheme as the Go2 / B2 / Airbox launchers: runs on Windows with Isaac Sim's own Python (standard
library only) and serves http://127.0.0.1:8790 . Only the fixed actions in ACTIONS can be started,
with parameters taken from fixed lists; nothing else can be executed through it. It listens on
127.0.0.1 only and every POST needs the page token (or the X-AHR header for shutdown from the .bat).

Start it with Open_launcher.bat (hidden) and stop it with Stop_launcher.bat or the Shut down button.
"""
import glob, json, os, re, secrets, subprocess, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('AHR_ROOT', os.path.dirname(HERE))
LOGS = os.path.join(ROOT, 'logs', 'launcher')
RL_LOGS = os.path.join(ROOT, 'logs', 'rsl_rl')
AGENT_RUNS = os.path.join(ROOT, 'agents', 'runs')
PORT = int(os.environ.get('AHR_LAUNCHER_PORT', '8790'))   # Go2 8765, B2 8770, Performance Lab 8090, Airbox 8780
DISTRO = os.environ.get('AHR_WSL_DISTRO', 'Ubuntu-24.04')
ISAACLAB = os.environ.get('AHR_ISAACLAB', r'C:\isaaclab')
ISAACLAB_BAT = os.path.join(ISAACLAB, 'isaaclab.bat')
OLLAMA = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
WROOT = '/mnt/c/arm_hand_robot'
TOKEN = secrets.token_hex(16)
NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
PY = sys.executable    # Isaac Sim's python.exe: also runs the reward agent (stdlib only)

WSL = ['wsl', '-d', DISTRO, '--']


def lab(*args):
    """isaaclab.bat with arguments, run from ROOT so training logs land in ROOT\\logs."""
    return ['cmd', '/c', ISAACLAB_BAT, *args]


def wsl_env(env, cmd):
    return WSL + ['bash', '-lc', f'source {WROOT}/scripts/wsl/activate.sh {env} && {cmd}']


# ---- choices offered by the page (every parameter must be one of these strings)
OWN_TASKS = ['ArmHand-Lift-Cube-SO101-v0', 'ArmHand-Lift-Cube-SO101-Quick-v0']
DISCOVER_KEYWORDS = ['ArmHand', 'SO101', 'Allegro', 'Shadow', 'Lift', 'Reach', 'Repose', 'Reorient']
ENVS = ['512', '256', '1024', '2048', '128', '64', '16']
ITERS = ['default', '2', '50', '300', '1500', '3000']
SEEDS = ['42', '1', '2', '3']
LLMS_DEFAULT = ['qwen2.5-coder:7b', 'qwen2.5:7b', 'llama3.1:8b', 'qwen2.5:3b']
PLAYGROUND_ENVS = ['LeapCubeRotateZAxis', 'LeapCubeReorient', 'PandaPickCube', 'PandaPickCubeOrientation']
TASKS_FILE = os.path.join(LOGS, 'tasks.json')
# Isaac Lab's "video" extra (moviepy<2) and its pure-Python helpers; installed with --no-deps on purpose
VIDEO_PKGS = ['moviepy==1.0.3', 'decorator==4.4.2', 'imageio>=2.5,<3', 'imageio-ffmpeg>=0.4', 'proglog<=1.0.0']


def discovered_tasks():
    try:
        with open(TASKS_FILE, encoding='utf-8') as f:
            data = json.load(f)
        return [t for t in data.get('tasks', []) if t not in OWN_TASKS]
    except (OSError, ValueError):
        return []


def all_tasks():
    return OWN_TASKS + discovered_tasks()


def ollama_models():
    try:
        with urllib.request.urlopen(OLLAMA + '/api/tags', timeout=1.5) as r:
            names = [m['name'] for m in json.loads(r.read()).get('models', [])]
        return {'up': True, 'models': names}
    except Exception:
        return {'up': False, 'models': []}


def llm_choices():
    info = cached('ollama', 10, ollama_models)
    have = info.get('models', []) if isinstance(info, dict) else []
    return have + [m for m in LLMS_DEFAULT if m not in have]


def override_files():
    """overrides.json of every reward-agent trial, newest session first, plus 'none'."""
    out = ['none']
    for path in sorted(glob.glob(os.path.join(AGENT_RUNS, '*', 'trial_*', 'overrides.json')), reverse=True)[:60]:
        out.append(os.path.relpath(path, AGENT_RUNS).replace('\\', '/'))
    return out


def train_cmd(p, action='train'):
    cmd = lab(action, '--rl_library', 'rsl_rl', '--task', p['task'], '--num_envs', p['num_envs'])
    if action == 'train':
        cmd += ['--seed', p['seed']]
        if p['iterations'] != 'default':
            cmd += ['--max_iterations', p['iterations']]
    if p.get('viz') == 'kit' or action == 'play':
        cmd += ['--viz', 'kit']
    return [cmd]


def job_env(p):
    """Extra environment for a job: reward overrides chosen on the page (path checked against the list)."""
    ov = p.get('overrides', 'none')
    if ov and ov != 'none':
        return {'ARM_HAND_OVERRIDES': os.path.join(AGENT_RUNS, *ov.split('/'))}
    return {}


# name -> (label, help, params {name: allowed values or callable}, builder(params) -> commands, needs_gpu)
ACTIONS = {
    'setup': ('Install task package', 'pip install -e sim\\arm_hand_lab into Isaac Sim\'s Python (no other changes), then list our tasks',
              {}, lambda p: [[PY, '-m', 'pip', 'install', '--no-deps', '-e', os.path.join(ROOT, 'sim', 'arm_hand_lab')],
                             lab('list_envs', '--keyword', 'ArmHand')], False),
    'install_video': ('Install video support', 'Adds MoviePy, imageio, imageio-ffmpeg, decorator and proglog to Isaac Sim\'s Python '
                      'with --no-deps, so numpy, pillow and other Isaac Sim packages are never installed or replaced '
                      '(inside Isaac Sim the bundled numpy/pillow are used). Needed by Record video',
                      {}, lambda p: [[PY, '-m', 'pip', 'install', '--no-deps', *VIDEO_PKGS]], False),
    'discover': ('Find tasks', 'Ask Isaac Lab for the arm and hand tasks (SO-101, Allegro, Shadow, lift, reach, reorient) '
                 'so they can be picked in Train (about 1 min)', {}, lambda p: [[PY, '-u', __file__, '--discover']], False),
    'smoke': ('Smoke test', 'Builds the SO-101 lift task with 64 envs and trains 2 iterations (1-3 min). First thing to run',
              {}, lambda p: train_cmd({'task': OWN_TASKS[0], 'num_envs': '64', 'seed': '42', 'iterations': '2'}), True),
    'train': ('Train', 'RSL-RL PPO training, headless unless the viewer is chosen. Checkpoints in logs\\rsl_rl',
              {'task': all_tasks, 'num_envs': ENVS, 'iterations': ITERS, 'seed': SEEDS, 'viz': ['none', 'kit'],
               'overrides': override_files},
              lambda p: train_cmd(p), True),
    'play': ('Play', 'Opens the Kit viewer with the latest checkpoint of the task (close the window or press Stop)',
             {'task': all_tasks, 'num_envs': ['16', '4', '64'], 'overrides': override_files},
             lambda p: train_cmd(p, 'play'), True),
    'video': ('Record video', 'Plays the latest checkpoint headless in 1 env and saves a 10 s mp4 next to it '
              '(Results tab). Lets you and Claude check what the policy really does (1-3 min)',
              {'task': all_tasks, 'overrides': override_files},
              lambda p: [lab('play', '--rl_library', 'rsl_rl', '--task', p['task'], '--num_envs', '1',
                             '--video', '--video_length', '200')], True),
    'agent': ('Reward agent', 'Local LLM (Ollama) proposes reward weights, each trial trains 300 iterations and is scored by '
              'Episode_Reward/success. 5-15 min per trial. Results under Local AI',
              {'trials': ['6', '3', '10', '1'], 'model': llm_choices, 'num_envs': ['512', '256', '1024']},
              lambda p: [[PY, '-u', os.path.join(ROOT, 'agents', 'reward_agent.py'), '--trials', p['trials'],
                          '--model', p['model'], '--num_envs', p['num_envs'], '--isaaclab_bat', ISAACLAB_BAT]], True),
    'ollama_pull': ('Download model', 'ollama pull of the chosen model (a few GB)', {'model': llm_choices},
                    lambda p: [['ollama', 'pull', p['model']]], False),
    'wsl_setup': ('WSL environments', 'Create ~/.venvs/ahr-lerobot (LeRobot, SmolVLA, PyTorch) and/or ~/.venvs/ahr-mujoco '
                  '(MuJoCo, Playground, JAX) in WSL. Does not touch ROS 2 or the system Python (10-20 min)',
                  {'target': ['all', 'lerobot', 'mujoco']},
                  lambda p: [WSL + ['bash', f'{WROOT}/scripts/wsl/setup_wsl.sh', p['target']]], False),
    'playground': ('Playground training', 'MuJoCo Playground PPO in WSL (JAX on the GPU). LEAP hand by default; '
                   'results in ~/ahr_data/playground', {'env': PLAYGROUND_ENVS, 'num_envs': ['2048', '1024', '4096', '512']},
                   lambda p: [wsl_env('mujoco', f'python {WROOT}/scripts/wsl/train_playground.py --env {p["env"]} --num-envs {p["num_envs"]}')], True),
    'gpu_check': ('GPU check', 'nvidia-smi on Windows and in WSL, plus the CUDA checks of the WSL environments that exist',
                  {}, lambda p: [['nvidia-smi'], WSL + ['bash', f'{WROOT}/scripts/wsl/gpu_check.sh']], False),
}

job = {'name': None, 'label': None, 'params': {}, 'proc': None, 'started': 0.0, 'ended': 0.0,
       'exit': None, 'log': None, 'step': '', 'stopping': False}
history = []
lock = threading.Lock()
cache = {}


def cached(key, ttl, fn):
    now = time.time()
    hit = cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        value = fn()
    except Exception as e:
        value = {'error': str(e)}
    cache[key] = (now, value)
    return value


def run_quiet(cmd, timeout=15):
    out = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=NO_WINDOW, stdin=subprocess.DEVNULL)
    return out.stdout.decode('utf-8', 'replace').strip()


def gpu_info():
    out = run_quiet(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw',
                     '--format=csv,noheader,nounits'], 8)
    parts = [x.strip() for x in out.splitlines()[0].split(',')] if out else []
    if len(parts) < 6:
        return {'text': out or 'nvidia-smi gave no answer'}
    return dict(zip(['name', 'vram_used_mb', 'vram_total_mb', 'temp_c', 'load_pct', 'power_w'], parts))


ISAAC_PS = (
    # kit.exe / isaac-sim.exe are always Isaac Sim (python.bat runs Isaac Lab as kit.exe). A python.exe only
    # counts when its ARGUMENTS (not the interpreter path) point at Isaac Sim / Isaac Lab: panels of other
    # projects that merely use Isaac Sim's python.exe (Airbox, B2/Go2 launchers) are not simulations.
    "Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'python.exe','kit.exe','isaac-sim.exe' } | ForEach-Object { "
    "$cl = '' + $_.CommandLine; $q = [char]34; $args_ = ($cl -replace ('^\\s*(' + $q + '[^' + $q + ']*' + $q + '|\\S+)\\s*'), ''); "
    "$isaac = ($_.Name -ne 'python.exe') -or ($args_ -match 'isaaclab|isaacsim|isaac-sim|isaac_sim|\\.kit\\b|SimulationApp'); "
    "if ($isaac -and ($args_ -notmatch '_launcher\\.py|panel\\.py')) { "
    "$c = ($args_ -replace [char]34, '') -split ' ' | Where-Object { $_ -match '\\.py$|\\.kit$|\\.bat$' } | Select-Object -First 1; "
    "if ($c) { $c = ('' + $c) -replace '^.*[\\\\/]', '' } else { $c = $args_.Substring(0, [Math]::Min(90, $args_.Length)) }; "
    "'' + $_.ProcessId + ' ' + $_.Name + ' ' + $c } }")


def isaac_processes():
    """Isaac Sim / Isaac Lab processes running on Windows (ours or another project's): ['<pid> <exe>']."""
    if os.name != 'nt':
        return []
    out = run_quiet(['powershell', '-NoProfile', '-NonInteractive', '-Command', ISAAC_PS], 20)
    return [line.strip() for line in out.splitlines() if line.strip()]


def descendants(pid):
    """PIDs started by the current job (so its own Isaac Sim is not reported as 'another' one)."""
    if os.name != 'nt' or not pid:
        return set()
    ps = ("$all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId; $ids = @(%d); $i = 0; "
          "while ($i -lt $ids.Count) { $p = $ids[$i]; $all | Where-Object { $_.ParentProcessId -eq $p } | "
          "ForEach-Object { $ids += $_.ProcessId }; $i++ }; $ids -join ' '" % int(pid))
    return set(run_quiet(['powershell', '-NoProfile', '-NonInteractive', '-Command', ps], 20).split())


def other_isaac():
    procs = cached('isaac', 15, isaac_processes)
    if not isinstance(procs, list):
        return []
    proc = job['proc']
    mine = cached('mine', 15, lambda: descendants(proc.pid)) if proc is not None and proc.poll() is None else set()
    return [p for p in procs if p.split()[0] not in mine]


# ---- progress of the running job, read from its log
ITER_RE = re.compile(r'Learning iteration\s+(\d+)\s*/\s*(\d+)')
METRIC_RE = re.compile(r'^\s*(Episode_Reward/success|Mean reward|Mean episode length|Episode_Termination/\w+)\s*:\s*(-?[\d.eE+-]+)', re.M)
TRIAL_RE = re.compile(r'=== Trial (\d+) ===')
PG_RE = re.compile(r'step\s+([\d,]+)\s+eval reward\s+(-?[\d.]+)')


def tail(path, n=60000):
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            if size > n:
                f.seek(size - n)
            return f.read().decode('utf-8', 'replace')
    except OSError:
        return ''


def progress(log_name):
    text = tail(os.path.join(LOGS, log_name)) if log_name else ''
    out = {}
    its = ITER_RE.findall(text)
    if its:
        out['iteration'], out['iterations'] = int(its[-1][0]), int(its[-1][1])
    for key, val in METRIC_RE.findall(text):
        out[key] = val
    trials = TRIAL_RE.findall(text)
    if trials:
        out['trial'] = int(trials[-1])
    pg = PG_RE.findall(text)
    if pg:
        out['pg_step'], out['pg_reward'] = pg[-1]
    # generic first, specific last: the most specific matching message wins
    for pat, msg in ((r'Traceback \(most recent call last\)', 'Python error: see the log'),
                     (r'CUDA out of memory|OutOfMemoryError|RESOURCE_EXHAUSTED', 'GPU out of memory: use fewer envs'),
                     (r'requires MoviePy', 'Video support missing: run Install video support')):
        if re.search(pat, text):
            out['warning'] = msg
    return out


# ---- job runner (one job at a time, same as the B2 launcher)
def worker(name, label, params, commands, log_path, extra_env):
    code = 0
    # UTF-8 output: Isaac Lab's banner uses characters that the Windows default (cp1252) cannot write to a file
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1', **extra_env)
    env.pop('VIRTUAL_ENV', None); env.pop('CONDA_PREFIX', None)   # isaaclab.bat refuses them with the bundled Python
    with open(log_path, 'w', encoding='utf-8', errors='replace') as log:
        if extra_env:
            log.write(''.join(f'{k}={v}\n' for k, v in extra_env.items()))
        for i, cmd in enumerate(commands):
            if job['stopping']:
                code = -1
                break
            job['step'] = f"{i + 1}/{len(commands)}: {' '.join(cmd)[:140]}"
            log.write(f"\n=== [{time.strftime('%H:%M:%S')}] {' '.join(cmd)}\n"); log.flush()
            try:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        cwd=ROOT, env=env, creationflags=NO_WINDOW)
            except OSError as e:
                log.write(f'could not start: {e}\n'); code = 127
                break
            job['proc'] = proc
            rc = proc.wait()
            if rc != 0:
                code = rc
                break
        if job['stopping'] and code == 0:
            code = -1
        log.write(f"\n=== [{time.strftime('%H:%M:%S')}] finished, exit code {code}\n")
    with lock:
        job.update(proc=None, ended=time.time(), exit=code, step='', stopping=False)
        history.insert(0, {'name': name, 'label': label, 'params': params, 'exit': code, 'started': job['started'],
                           'seconds': round(job['ended'] - job['started']), 'log': os.path.basename(log_path),
                           'result': progress(os.path.basename(log_path))})
        del history[40:]
        save_history()
    cache.pop('runs', None); cache.pop('agent', None)


def save_history():
    try:
        with open(os.path.join(LOGS, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f)
    except OSError:
        pass


def load_history():
    try:
        with open(os.path.join(LOGS, 'history.json'), encoding='utf-8') as f:
            history.extend(json.load(f)[:40])
    except (OSError, ValueError):
        pass


def allowed_values(spec):
    return spec() if callable(spec) else spec


def start(name, params):
    if name not in ACTIONS:
        return 400, {'error': 'unknown action'}
    label, _, allowed, builder, needs_gpu = ACTIONS[name]
    clean = {}
    for key, spec in allowed.items():
        values = allowed_values(spec)
        value = str(params.get(key, values[0] if values else ''))
        if value not in values:
            return 400, {'error': f'bad value for {key}: {value}'}
        clean[key] = value
    if needs_gpu and params.get('ignore_other') is not True:
        others = other_isaac()
        if others:
            return 409, {'error': 'Another Isaac Sim is running (' + ', '.join(others) + '). Close it, or tick '
                                  '"Run anyway" (8 GB VRAM: both may run out of memory).', 'other': others}
    with lock:
        if job['name'] and not job['ended']:
            return 409, {'error': f"'{job['label']}' is still running"}
        os.makedirs(LOGS, exist_ok=True)
        log_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{name}.txt"
        job.update(name=name, label=label, params=clean, started=time.time(), ended=0.0, exit=None,
                   log=log_name, step='starting', stopping=False)
    prune_logs()
    threading.Thread(target=worker, args=(name, label, clean, builder(clean), os.path.join(LOGS, log_name), job_env(clean)),
                     daemon=True).start()
    return 200, {'ok': True, 'log': log_name}


def prune_logs(keep=80):
    files = sorted(glob.glob(os.path.join(LOGS, '2*_*.txt')))
    for f in files[:-keep]:
        try:
            os.remove(f)
        except OSError:
            pass


def kill_tree(proc):
    if proc is None or proc.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True, creationflags=NO_WINDOW)
    else:
        proc.kill()


def stop():
    """Stop the running job and everything it started. Never touches other projects' Isaac Sim or ROS 2."""
    name = job['name']
    job['stopping'] = True
    kill_tree(job['proc'])
    if name == 'playground':   # the Python inside WSL is not a Windows child process
        try:
            subprocess.run(WSL + ['pkill', '-f', 'train_playground.py'], timeout=20, creationflags=NO_WINDOW,
                           capture_output=True)
        except Exception:
            pass
    return 200, {'ok': True}


# ---- results
def runs():
    """Training runs under logs\\rsl_rl: experiment, run, checkpoints, newest first."""
    out = []
    for run_dir in glob.glob(os.path.join(RL_LOGS, '*', '*')):
        if not os.path.isdir(run_dir):
            continue
        ckpts = glob.glob(os.path.join(run_dir, 'model_*.pt'))
        nums = sorted(int(m.group(1)) for c in ckpts for m in [re.search(r'model_(\d+)\.pt$', c)] if m)
        videos = sorted(glob.glob(os.path.join(run_dir, 'videos', '*', '*.mp4')), key=os.path.getmtime, reverse=True)
        out.append({'experiment': os.path.basename(os.path.dirname(run_dir)), 'run': os.path.basename(run_dir),
                    'videos': [os.path.relpath(v, RL_LOGS).replace('\\', '/') for v in videos[:6]],
                    'checkpoints': len(nums), 'last': nums[-1] if nums else None, 'mtime': os.path.getmtime(run_dir)})
    out.sort(key=lambda r: -r['mtime'])
    return out[:40]


def agent_sessions():
    out = []
    for hist in sorted(glob.glob(os.path.join(AGENT_RUNS, '*', 'history.jsonl')), reverse=True)[:15]:
        trials = []
        try:
            with open(hist, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        h = json.loads(line)
                        trials.append({'trial': h.get('trial'), 'score': h.get('score'), 'weights': h.get('weights'),
                                       'analysis': (h.get('analysis') or '')[:400], 'exit': h.get('exit_code'),
                                       'metrics': {k: v for k, v in (h.get('metrics') or {}).items() if k.startswith('Episode_Reward')}})
        except (OSError, ValueError):
            pass
        best = max(trials, key=lambda t: t['score'] or 0) if trials else None
        out.append({'session': os.path.basename(os.path.dirname(hist)), 'trials': trials,
                    'best': best['trial'] if best else None, 'best_score': best['score'] if best else None})
    return out


def status():
    with lock:
        j = {k: job[k] for k in ('name', 'label', 'params', 'started', 'ended', 'exit', 'log', 'step')}
        j['running'] = job['name'] is not None and not job['ended']
        j['seconds'] = round((job['ended'] or time.time()) - job['started']) if job['started'] else 0
        hist = list(history)
    j['progress'] = progress(j['log']) if j['log'] else {}
    logs = []
    try:
        for f in os.listdir(LOGS):
            p = os.path.join(LOGS, f)
            if os.path.isfile(p) and f.endswith('.txt'):
                logs.append({'name': f, 'mtime': os.path.getmtime(p), 'size': os.path.getsize(p)})
    except OSError:
        pass
    logs.sort(key=lambda x: -x['mtime'])
    actions = {}
    for k, (label, help_, params, _, gpu) in ACTIONS.items():
        actions[k] = {'label': label, 'help': help_, 'gpu': gpu,
                      'params': {pk: allowed_values(pv) for pk, pv in params.items()}}
    return {'version': 1, 'job': j, 'history': hist, 'gpu': cached('gpu', 5, gpu_info),
            'ollama': cached('ollama', 10, ollama_models), 'other_isaac': other_isaac(),
            'tasks_discovered': os.path.isfile(TASKS_FILE), 'runs': cached('runs', 10, runs),
            'agent': cached('agent', 10, agent_sessions), 'logs': logs[:60], 'actions': actions,
            'root': ROOT, 'time': time.time()}


def report():
    """Plain-text summary for Claude (also shown on the page): state, progress, last lines of the log."""
    s = status()
    j = s['job']
    lines = [f"Arm Hand Robot launcher @ {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"job: {j['label'] or '-'} {json.dumps(j['params'])} running={j['running']} exit={j['exit']} "
             f"seconds={j['seconds']} log={j['log']}",
             f"progress: {json.dumps(j['progress'])}",
             f"gpu: {json.dumps(s['gpu'])}", f"ollama: up={s['ollama'].get('up')} models={s['ollama'].get('models')}",
             f"other Isaac Sim: {s['other_isaac'] or 'none'}", 'recent jobs:']
    for h in s['history'][:8]:
        lines.append(f"  {h['label']} {json.dumps(h['params'])} exit={h['exit']} {h['seconds']}s result={json.dumps(h.get('result', {}))}")
    if j['log']:
        text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', tail(os.path.join(LOGS, j['log']), 5000))
        lines += ['', f'--- last lines of {j["log"]} ---', *text.splitlines()[-50:]]
    return '\n'.join(lines)


ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
SAFE = re.compile(r'^[\w.\-]+$')
VIDEO_FILE = re.compile(r'^[\w.\-]+/[\w.\-]+/videos/[\w.\-]+/[\w.\-]+\.mp4$')
AGENT_FILE = re.compile(r'^\d{8}_\d{6}/(summary\.md|best\.json|history\.jsonl|trial_\d{2}/(overrides\.json|train\.log))$')


def read_log(name, n):
    if not SAFE.match(name or ''):
        return None
    path = os.path.realpath(os.path.join(LOGS, name))
    if not path.startswith(os.path.realpath(LOGS) + os.sep) or not os.path.isfile(path):
        return None
    return ANSI.sub('', tail(path, n) if n else open(path, encoding='utf-8', errors='replace').read())


def read_agent_file(rel, n=200000):
    if not AGENT_FILE.match(rel or ''):
        return None
    path = os.path.realpath(os.path.join(AGENT_RUNS, *rel.split('/')))
    if not path.startswith(os.path.realpath(AGENT_RUNS) + os.sep) or not os.path.isfile(path):
        return None
    return ANSI.sub('', tail(path, n))


class Handler(BaseHTTPRequestHandler):
    server_version = 'AhrLauncher/1'

    def log_message(self, *args):
        pass

    def send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def host_ok(self):
        return (self.headers.get('Host') or '').split(':')[0] in ('127.0.0.1', 'localhost')

    def do_GET(self):
        if not self.host_ok():
            return self.send(403, {'error': 'forbidden'})
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path in ('/', '/index.html'):
            with open(os.path.join(HERE, 'index.html'), encoding='utf-8') as f:
                return self.send(200, f.read().replace('__TOKEN__', TOKEN), 'text/html; charset=utf-8')
        if url.path == '/api/status':
            return self.send(200, status())
        if url.path == '/api/report':
            return self.send(200, report(), 'text/plain; charset=utf-8')
        if url.path == '/api/log':
            text = read_log(q.get('name', [''])[0], int(q.get('tail', ['40000'])[0] or 0))
            return self.send(200, text, 'text/plain; charset=utf-8') if text is not None else self.send(404, {'error': 'not found'})
        if url.path == '/api/video':
            rel = q.get('file', [''])[0]
            if not VIDEO_FILE.match(rel or '') or '..' in rel:
                return self.send(404, {'error': 'not found'})
            path = os.path.realpath(os.path.join(RL_LOGS, *rel.split('/')))
            if not path.startswith(os.path.realpath(RL_LOGS) + os.sep) or not os.path.isfile(path):
                return self.send(404, {'error': 'not found'})
            with open(path, 'rb') as f:
                return self.send(200, f.read(), 'video/mp4')
        if url.path == '/api/agent_file':
            text = read_agent_file(q.get('file', [''])[0])
            return self.send(200, text, 'text/plain; charset=utf-8') if text is not None else self.send(404, {'error': 'not found'})
        return self.send(404, {'error': 'not found'})

    def do_POST(self):
        if not self.host_ok():
            return self.send(403, {'error': 'forbidden'})
        if self.path == '/api/shutdown' and (self.headers.get('X-AHR') == '1' or self.headers.get('X-AHR-Token') == TOKEN):
            stop()
            self.send(200, {'ok': True})
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0)), daemon=True).start()
            return
        if self.headers.get('X-AHR-Token') != TOKEN:
            return self.send(403, {'error': 'forbidden'})
        length = int(self.headers.get('Content-Length') or 0)
        try:
            body = json.loads(self.rfile.read(length) or b'{}')
        except ValueError:
            return self.send(400, {'error': 'bad json'})
        if self.path == '/api/run':
            return self.send(*start(str(body.get('action', '')), body.get('params') or {}))
        if self.path == '/api/stop':
            return self.send(*stop())
        if self.path == '/api/reload':
            if job['name'] and not job['ended']:
                return self.send(409, {'error': f"'{job['label']}' is still running"})
            self.send(200, {'ok': True})
            threading.Thread(target=reload_launcher, daemon=True).start()
            return
        return self.send(404, {'error': 'not found'})


SERVER = None


def reload_launcher():
    time.sleep(0.5)
    try:
        SERVER.server_close()
    except Exception:
        pass
    subprocess.Popen([sys.executable] + sys.argv, cwd=ROOT, creationflags=NO_WINDOW)
    os._exit(0)


def discover():
    """--discover: run `isaaclab.bat list_envs --keyword K` for each keyword and save the task IDs."""
    found = []
    for kw in DISCOVER_KEYWORDS:
        print(f'list_envs --keyword {kw}', flush=True)
        try:
            out = subprocess.run(lab('list_envs', '--keyword', kw), capture_output=True, timeout=600, cwd=ROOT,
                                 creationflags=NO_WINDOW, stdin=subprocess.DEVNULL).stdout.decode('utf-8', 'replace')
        except Exception as e:
            print(f'  failed: {e}')
            continue
        ids = re.findall(r'^\|\s*\d+\s*\|\s*([\w.\-]+)\s*\|', out, re.M)
        print('  ' + (', '.join(ids) or 'none'), flush=True)
        found += [i for i in ids if i not in found]
    os.makedirs(LOGS, exist_ok=True)
    with open(TASKS_FILE, 'w', encoding='utf-8') as f:
        json.dump({'time': time.time(), 'tasks': found}, f, indent=1)
    print(f'{len(found)} tasks saved to {TASKS_FILE}')


def main():
    global SERVER
    if '--discover' in sys.argv:
        return discover()
    os.makedirs(LOGS, exist_ok=True)
    load_history()
    for _ in range(20):   # the port may still be held for a moment by a launcher that is being replaced
        try:
            SERVER = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
            break
        except OSError:
            time.sleep(0.5)
    if SERVER is None:
        sys.exit(f'port {PORT} is in use: is another launcher running?')
    print(f'Arm Hand Robot launcher on http://127.0.0.1:{PORT}  (root {ROOT})', flush=True)
    try:
        SERVER.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
