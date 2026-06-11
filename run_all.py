import subprocess

instances = [
    'instancias/A-n32-k5.vrp',
    'instancias/A-n33-k5.vrp',
    'instancias/A-n33-k6.vrp',
    'instancias/A-n34-k5.vrp',
    'instancias/A-n36-k5.vrp',
]

toy = 'instancias/toy_problem.vrp'

seeds = [i for i in range(0, 10)]

subprocess.run(
    ["python3", "run_brkga.py", toy]
    + ["--config", "configs/config_toy.conf"]
    + ["--seeds"] + [str(s) for s in seeds]
    + ["--max_time", "5"]
)

for instance in instances:
    subprocess.run(
        ["python3", "run_brkga.py", instance, "--seeds"]
        + [str(s) for s in seeds]
        + ["--max_time", "450"]
    )
