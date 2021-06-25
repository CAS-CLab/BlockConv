import logging
import logging.config
import time
import os
import contextlib
import platform 
import shutil
import sys
from git import Repo, InvalidGitRepositoryError
import numpy as np
import torch
try:
    import lsb_release
    HAVE_LSB = True
except ImportError:
    HAVE_LSB = False

logger = logging.getLogger("app_cfg")

def config_logger(log_cfg_file, experiment_name, output_dir='logs'):
    timestr = time.strftime("%Y.%m.%d-%H%M%S")
    exp_full_name = timestr if experiment_name is None else experiment_name + '_' + timestr
    logdir = os.path.join(output_dir, exp_full_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_filename = os.path.join(logdir, exp_full_name + '.log')
    if os.path.isfile(log_cfg_file):
        logging.config.fileConfig(log_cfg_file, defaults={'logfilename':log_filename})

    msglogger = logging.getLogger()
    msglogger.logdir = logdir
    msglogger.log_filename = log_filename
    msglogger.info('Log file for this run: ' + os.path.realpath(log_filename))

    return msglogger, logdir

def log_execution_env_state(logdir=None, gitroot='.'):
    """Log information about the execution environment.

    File 'config_path' will be copied to directory 'logdir'. A common use-case
    is passing the path to a (compression) schedule YAML file. Storing a copy
    of the schedule file, with the experiment logs, is useful in order to
    reproduce experiments.

    Args:
        config_path: path to config file, used only when logdir is set
        logdir: log directory
        git_root: the path to the .git root directory
    """

    def log_git_state():
        """Log the state of the git repository.

        It is useful to know what git tag we're using, and if we have outstanding code.
        """
        try:
            repo = Repo(gitroot)
            assert not repo.bare
        except InvalidGitRepositoryError:
            logger.debug("Cannot find a Git repository.  You probably downloaded an archive of Distiller.")
            return

        if repo.is_dirty():
            logger.debug("Git is dirty")
        try:
            branch_name = repo.active_branch.name
        except TypeError:
            branch_name = "None, Git is in 'detached HEAD' state"
        logger.debug("Active Git branch: %s", branch_name)
        logger.debug("Git commit: %s" % repo.head.commit.hexsha)

    logger.debug("Number of CPUs: %d", len(os.sched_getaffinity(0)))
    logger.debug("Number of GPUs: %d", torch.cuda.device_count())
    logger.debug("CUDA version: %s", torch.version.cuda)
    logger.debug("CUDNN version: %s", torch.backends.cudnn.version())
    logger.debug("Kernel: %s", platform.release())
    if HAVE_LSB:
        logger.debug("OS: %s", lsb_release.get_lsb_information()['DESCRIPTION'])
    logger.debug("Python: %s", sys.version)
    logger.debug("PyTorch: %s", torch.__version__)
    logger.debug("Numpy: %s", np.__version__)
    log_git_state()
    logger.debug("Command line: %s", " ".join(sys.argv))