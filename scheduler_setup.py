import os
import platform
import subprocess


def schedule_cron_job():
    script_path = os.path.abspath("run_batch.py")
    log_path = os.path.abspath("batch_log.log")
    cron_job = f"0 2 * * * /usr/bin/python3 {script_path} >> {log_path} 2>&1"

    # Check if the job already exists
    existing_cron_jobs = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    if cron_job not in existing_cron_jobs:
        new_cron_jobs = existing_cron_jobs + cron_job + "\n"
        subprocess.run(["crontab"], input=new_cron_jobs, text=True)
        print("Cron job scheduled successfully.")
    else:
        print("Cron job already exists.")


def schedule_windows_task():
    script_path = os.path.abspath("run_batch.py")
    task_name = "RunBatchPrediction"

    # Remove existing task if any
    subprocess.run(["schtasks", "/Delete", "/TN", task_name, "/F"], capture_output=True, text=True)

    # Create new scheduled task
    command = (
        f'schtasks /Create /SC DAILY /TN {task_name} /TR "python {script_path}" /ST 02:00'
    )
    subprocess.run(command, shell=True)
    print("Windows Task Scheduler job created successfully.")


def main():
    os_type = platform.system()
    if os_type == "Linux" or os_type == "Darwin":
        schedule_cron_job()
    elif os_type == "Windows":
        schedule_windows_task()
    else:
        print("Unsupported OS detected. Please set up scheduling manually.")


if __name__ == "__main__":
    main()