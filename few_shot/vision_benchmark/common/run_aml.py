import logging
import os
import pathlib
import shutil
import tempfile
import azureml.core

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

EXPERIMENT_NAME = 'vision-benchmark'
AZUREML_CLUSTER = os.getenv('AZUREML_CLUSTER')
AZUREML_WORKSPACE = os.getenv('AZUREML_WORKSPACE')
AZUREML_SUBSCRIPTION_ID = os.getenv('AZUREML_SUBSCRIPTION_ID')


class AMLRunner:
    def __init__(self):
        self.run = None

        self.temp_dir = tempfile.mkdtemp()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.temp_dir)

    def add_input(self, local_filepath):
        returns_list = True
        if not isinstance(local_filepath, list):
            local_filepath = [local_filepath]
            returns_list = False

        filepaths = []
        for lf in local_filepath:
            filename = os.path.basename(os.path.normpath(lf))
            if os.path.exists(os.path.join(self.temp_dir, filename)):
                raise RuntimeError(f'Duplicated input filepath: {filename}')

            if os.path.isdir(lf):
                shutil.copytree(lf, os.path.join(self.temp_dir, filename))
            else:
                shutil.copy(lf, self.temp_dir)

            filepaths.append(filename)
        return filepaths if returns_list else filepaths[0]

    def download_output_file(self, remote_filepath, local_filepath):
        if not self.run:
            raise RuntimeError("Need to submit a job first.")
        self.run.download_file(remote_filepath, local_filepath)

    def run_job(self, command, experiment, compute_target):
        self._prepare_for_run('main.py', command)

        # Construct the run config and submit.
        run_config = self._construct_run_config(compute_target)
        script_run_config = azureml.core.ScriptRunConfig(source_directory=self.temp_dir, script='main.py', run_config=run_config)
        self.run = experiment.submit(config=script_run_config)
        return self.run

    def wait_for_job(self):
        self.run.wait_for_completion(show_output=True)

    def _prepare_for_run(self, entry_script_name, command):
        # Copy the local code directory.
        code_source_dir = pathlib.Path(__file__).resolve().parents[1]
        setup_py_filepath = code_source_dir.parent / 'setup.py'

        shutil.copytree(code_source_dir, os.path.join(self.temp_dir, os.path.basename(code_source_dir)))
        shutil.copy(setup_py_filepath, self.temp_dir)
        pkg_install_cmd = 'pip install -e .'
        pkg_install_cmd2 = 'pip install git+https://github.com/openai/CLIP.git'

        # Make an entrypoint script.
        with open(os.path.join(self.temp_dir, entry_script_name), 'w') as f:
            f.write('import os\n')
            f.write('import subprocess\n')
            f.write('import sys\n')
            f.write('import time\n')

            f.write(f'return_code = os.system({repr(pkg_install_cmd)})\n')
            f.write('print(f"Return code: {return_code}")\n')
            f.write(f'return_code = os.system({repr(pkg_install_cmd2)})\n')
            f.write('print(f"Return code: {return_code}")\n')

            f.write(f'return_code = os.system({repr(command)})\n')
            f.write('print(f"Return code: {return_code}")\n')
            f.write('time.sleep(20)\n')  # Wait for 20s to work around the issue AML cannot upload the output files.
            f.write('sys.exit(return_code)')

    @staticmethod
    def _construct_run_config(compute_target):
        run_config = azureml.core.runconfig.RunConfiguration()
        if compute_target:
            run_config.target = compute_target
        dependencies = azureml.core.conda_dependencies.CondaDependencies()
        dependencies.set_python_version('3.7')
        # The libraries installed in this step will be cached into the docker image, so it can save time/network on runtime.
        # This is based on an assumption that library dependency is rarely changed.
        # It will run "pip install -e" inside a docker container to use the local modified irispy.
        # dependencies.add_pip_package('irispy[run]')
        # dependencies.set_pip_option('--extra-index-url ' + self.pip_index)
        run_config.environment.python.conda_dependencies = dependencies
        run_config.environment.docker.base_image = 'mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'

        # Increase the shared memory size from the default 2GB.
        # The shared memory is used in PyTorch DataLoader to send processed batches to the main thread.
        # It holds up to 2 * num_workers batches. When an image size is 1000x1000 and batch size is 64, the size of a batch is (1000 * 1000 * 3 * 4) * 64 = 768MB.
        # When num_workers=4, the max shared memory usage should be 768 * 4 * 2 = 6144MB.
        # However, 8GB seems to be not enough in some cases. Not sure why.
        run_config.docker = azureml.core.runconfig.DockerConfiguration(use_docker=True, shm_size='16g')
        return run_config


def _get_workspace():
    """Get AML Workspace. As of 2021/08/20, Workspace.get() doesn't work due to a bug."""

    logger.info('Use interactive authentication...')

    workspaces_dict = azureml.core.Workspace.list(subscription_id=AZUREML_SUBSCRIPTION_ID, auth=None)
    workspaces = workspaces_dict.get(AZUREML_WORKSPACE)
    if not workspaces:
        raise RuntimeError(f'Workspace {AZUREML_WORKSPACE} not found')
    if len(workspaces) >= 2:
        raise RuntimeError(f'Multiple workspaces were found. len(workspaces) = {len(workspaces)}')
    logger.info(f'Workspace: {workspaces[0]}')

    return workspaces[0]


def run_aml(args, construct_cmd, track):
    # This is a list of arguments which specify input files or directories.
    potential_input_args = ['ds', 'model']
    with AMLRunner() as runner:
        for input_arg_name in potential_input_args:
            if hasattr(args, input_arg_name):
                filepath = getattr(args, input_arg_name)

                def process_single_file(f):
                    logger.info(f"Adding an input file/directory: {input_arg_name} {f}")
                    return runner.add_input(f)

                if filepath:
                    if isinstance(filepath, list):
                        filepath = [process_single_file(f) for f in filepath]
                    else:
                        filepath = process_single_file(filepath)

                    setattr(args, input_arg_name, filepath)
        cmd = construct_cmd(args)
        flatten_cmd = []
        for x in cmd:
            if isinstance(x, list):
                flatten_cmd.extend(x)
            else:
                flatten_cmd.append(x)

        command = ' '.join(flatten_cmd)

        logger.info(f'Constructed command: {command}')

        workspace = _get_workspace()
        experiment = azureml.core.Experiment(workspace=workspace, name=EXPERIMENT_NAME)
        cluster = azureml.core.compute.ComputeTarget(workspace=workspace, name=AZUREML_CLUSTER)
        if not cluster:
            raise RuntimeError(f'Cluster {AZUREML_CLUSTER} does not exist in workspace {AZUREML_WORKSPACE}')
        run = runner.run_job(command, experiment, cluster)

        run.tag('Data', args.ds)
        run.tag('Model', args.model)
        run.tag('Track', track)

        logger.info(f'RunId: {run.id}')
        logger.info(f'Web view: {run.get_portal_url()}')

        return run
