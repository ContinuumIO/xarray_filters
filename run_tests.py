import yaml
import glob
import subprocess
import sys
import time
import datetime
import shlex

assert sys.version_info >= (3,)

# Run one test per conda env file

def utc_timestamp():
    return datetime.datetime.timestamp(datetime.datetime.utcnow())

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
    cmd_output = lambda cmd: subprocess.check_output(shlex.split(cmd)).decode('utf-8').rstrip()
    test_report = {
        'git_commit': cmd_output('git rev-parse HEAD'),
        'git_diff': cmd_output('git diff'),
        'git_remotes': cmd_output('git remote -v'),
        'git_user.name': cmd_output('git config user.name'),
        'git_user.email': cmd_output('git config user.email'),
        'conda_env_spec.yaml': cmd_output('conda env export -n ' + env_name)
    }
    test_report['t0_UTC'] = utc_timestamp()
    test_report['test_report'] = run_tests(env_name)
    test_report['t1_UTC'] = utc_timestamp()
    test_report['test_status'] = [l.rstrip()
            for l in test_report['test_report'].splitlines()
            if (l.startswith('=============') and 'seconds' in l)][-1]
    return test_report

if __name__ == "__main__":
    import os
    import json

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
        print("Running tests for environment {:d} out of {:d}: {}".format(
            testn + 1, ntests, env_data['name']))
        if not exists_env(env_data['name']):
            create_env(enf_fpath)
        test_report = make_test_report(env_data['name'])
        timestamp_suffix = datetime.datetime.fromtimestamp(test_report['t0_UTC']).strftime('%Y%m%d_%H%M%S')
        test_log_fname = env_data['name'] + '_test-' + timestamp_suffix + '.json'
        report_path = os.path.join(test_logs_dir, test_log_fname)
        with open(report_path, mode='a') as f:
            json.dump(test_report, f, indent=4)
        print('Test log in {}'.format(report_path))
        print(test_report['test_status'] + '\n')

    print("All tests finished. Run\n")
    print("    echo -e $(jq '.test_report' <logfile_path>)")
    print("\nto see the test report. Additional metadata stored in the log file json.")

