from slurm_launcher.sbatch_launcher import launch_tasks


def run_exp():
    base_cmd = "python -B train.py"  # flexible one
    param_dict = {
        "--seed": range(10),
    }
    job_name = "twm-model-free"

    launch_tasks(
        param_option=1,
        base_cmd=base_cmd,
        param_dict=param_dict,
        partition="rtx3090",
        # exclude="mango,kalman,john,lemon,kiwi,xavier,zealot,apple",
        timeout="7-00:00:00",
        job_name=job_name,
    )


if __name__ == "__main__":
    run_exp()
