# ./run_experiment.sh trial unstructured --method dg_only --headless
# ./run_experiment.sh trial structured --method dg_only --headless
# ./run_experiment.sh trial unstructured --method vlm_only --headless
# ./run_experiment.sh trial structured --method vlm_only --headless
# ./run_experiment.sh trial unstructured --method dg_into_vlm --headless
# ./run_experiment.sh trial structured --method dg_into_vlm --headless
# ./run_experiment.sh trial unstructured structured --method dg_only vlm_dg --headless --base-dir docker-run-1
# ./run_experiment.sh trial short_test --method random --server --rosbag


# ./run_experiment.sh trial short_test --method dg_only --server --rosbag --mj-pickle --profile-grasping
# ./run_experiment.sh trial short_test --method vlm_only --rosbag --mj-pickle

# ./run_experiment.sh trial short_test --method all_auto --rosbag --mj-pickle
# ./run_experiment.sh resume_trial experiments/runs/trial_2026-03-04_08-39-13__short_test__all_auto --server --rerun-error

# ./run_experiment.sh resume_trial experiments/runs/trial_2026-03-02_02-20-07__all__human --server --rerun-error