import yaml
import glob
import subprocess
import sys
import time
import logging



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
    cmd = 'source activate {} && pytest --cov && source deactivate'.format(env_name)
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        retcode = proc.poll()
        out = proc.stdout.read()
    return retcode, out.decode(encoding='utf-8')

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
    test_report['t0_UTC_ms'] = int(time.time() * 1000)
    test_report['retcode'], test_report['report'] = run_tests(env_name)
    test_report['t1_UTC_ms'] = int(time.time() * 1000)
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
        print("Running test {:d} out of {:d}".format(testn + 1, ntests))
        print("Environment {} ".format(env_data['name']))
        time.sleep(0.5)
        if not exists_env(env_data['name']):
            create_env(enf_fpath)
        test_report = make_test_report(env_data['name'])
        test_log_fname = '_'.join([env_data['name'], str(test_report['t0_UTC_ms']), 'test.log'])
        report_path = os.path.join(test_logs_dir, test_log_fname)
        with open(report_path, mode='w') as f:
            json.dump(test_report, f, indent=4)
        t0, t1, retcode = [test_report.get(x) for x in ('t0_UTC_ms', 't1_UTC_ms', 'retcode')]
        print('DONE. Test finished in {:f} seconds with return code {}'.format(
            (t1 - t0) / 1000., str(retcode)))
        print('Test log in {}\n'.format(report_path))
    print("\nRun\n")
    print("    echo -e $(jq '.report' test_logs/<logfile>)")
    print("\nto see the test report. Additional metadata stored in the json.\n\n")

