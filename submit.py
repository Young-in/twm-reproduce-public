from slurm_launcher.sbatch_launcher import launch_tasks


def run_exp():
    base_cmd = "python -B train.py"  # flexible one
    param_dict = {
        "--seed": range(1),
    }
    job_name = "twm-model-based"

    launch_tasks(
        param_option=1,
        base_cmd=base_cmd,
        param_dict=param_dict,
        partition="a100",
        timeout="4-00:00:00",
        job_name=job_name,
    )


if __name__ == "__main__":
    run_exp()
