import os, glob
import yaml


def remove_runs_without_checkpoints(source):
    runs = glob.glob(source + '*/*/')
    num_cps = [len(glob.glob(x+'checkpoints/*')) for x in runs]

    for i, run in enumerate(runs):
        if num_cps[i] == 0:
            os.system('rm -rf ' + run)


source = '/media/ExtHDD01/logs/FlyZ/'
runs = glob.glob(source + '*/*/')

remove_runs_without_checkpoints(source)

runs = glob.glob(source + '*/*/')
for run in runs:
    with open(run + 'hparams.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        prj_name = data['prj']
        print(prj_name)
    os.system('mv ' + run + ' ' + run.replace(run.split('/')[-2], prj_name))