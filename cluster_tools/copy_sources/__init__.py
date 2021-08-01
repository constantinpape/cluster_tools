from .copy_sources import CopySourcesLocal, CopySourcesSlurm, CopySourcesLSF


def get_copy_task(target):
    if target == "local":
        return CopySourcesLocal
    elif target == "slurm":
        return CopySourcesSlurm
    elif target == "lsf":
        return CopySourcesLSF
    else:
        raise ValueError(f"Target {target} is not supported")
