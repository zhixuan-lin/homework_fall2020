# Notes

## Commands

```bash
python cs285/scripts/run_hw1.py \
--eval_batch_size 50000 \
--num_agent_train_steps_per_iter 10000 \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Ant-v2.pkl \
--video_log_freq -1

python cs285/scripts/run_hw1.py \
--eval_batch_size 50000 \
--num_agent_train_steps_per_iter 10000 \
--expert_policy_file cs285/policies/experts/Hopper.pkl \
--env_name Hopper-v2 --exp_name bc_hopper --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Hopper-v2.pkl \
--video_log_freq -1

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1 \
--eval_batch_size 5000 \
--num_agent_train_steps_per_iter 50000 \
--expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl \
--video_log_freq -1

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name dagger_humanoid --n_iter 100 \
--eval_batch_size 5000 \
--num_agent_train_steps_per_iter 500 \
--do_dagger --expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl \
--video_log_freq -1
```