import yaml
import glob
import subprocess
import sys
import time
import logging
import datetime

assert sys.version_info >= (3,)

# Run one test per conda env file

def exists_env(env_name):
    """Check if the environment name from env_fpath exists."""
    cmd = 'conda env list'
    out = subprocess.check_output(cmd, shell=True)
    out_lines = out.decode('utf-8').splitlines()
    env_names = [l.split()[0] for l in out_lines if l]
    env_exists = env_name in env_names
    return env_exists

def create_env(env_fpath):
    """Make conda env create command based on environment*.yml file"""
    cmd = 'conda env create -f {}'.format(env_fpath)
    out = subprocess.check_output(cmd, shell=True)
    return out

def run_tests(env_name):
    """Run tests in the conda env of env_fpath."""
    test_cmd = 'pytest --cov'
    cmd = 'source activate {} && {} && source deactivate'.format(env_name, test_cmd)
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        out = proc.stdout.read()
    return out.decode(encoding='utf-8')

def make_test_report(env_name):
    test_report = {
        'git_commit': subprocess.check_output('git rev-parse HEAD',
            shell=True).decode('utf-8').rstrip(),
        'git_diff': subprocess.check_output('git diff',
            shell=True).decode('utf-8').rstrip(),
        'git_remotes': subprocess.check_output('git remote -v',
            shell=True).decode('utf-8').rstrip(),
        'git_user.name': subprocess.check_output('git config user.name',
            shell=True).decode('utf-8').rstrip(),
        'git_user.email': subprocess.check_output('git config user.email',
            shell=True).decode('utf-8').rstrip()
    }
    test_report['t0_UTC'] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
    test_report['report'] = run_tests(env_name)
    test_report['t1_UTC'] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
    test_report['status'] = [l.rstrip()
            for l in test_report['report'].splitlines()
            if (l.startswith('=============') and 'seconds' in l)][-1]
    return test_report

if __name__ == "__main__":
    import os
    import json

    logging.basicConfig()
    test_logs_dir = 'test_logs'

    try:
        os.mkdir(test_logs_dir)
    except FileExistsError:
        pass

    env_files = glob.glob('environment*.yml') + glob.glob('environment*.yaml')
    ntests = len(env_files)

    for (testn, env_fpath) in enumerate(env_files):
        with open(env_fpath) as f:
            env_data = yaml.load(f)
        print("Running test {:d} out of {:d}. Environment: {}".format(
            testn + 1, ntests, env_data['name']))
        if not exists_env(env_data['name']):
            create_env(enf_fpath)
        test_report = make_test_report(env_data['name'])
        test_log_fname = '_'.join([env_data['name'], str(test_report['t0_UTC']), 'test.json'])
        report_path = os.path.join(test_logs_dir, test_log_fname)
        with open(report_path, mode='w') as f:
            json.dump(test_report, f, indent=4)
        t0, t1, retcode = [test_report.get(x) for x in ('t0_UTC', 't1_UTC', 'retcode')]
        print('Test log in {}'.format(report_path))
        print(test_report['status'] + '\n')
    print("All tests finished. Run\n")
    print("    echo -e $(jq '.report' <logfile_path>)")
    print("\nto see the test report. Additional metadata stored in the log file json.")

