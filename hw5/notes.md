# HW5

The code in this homework doesn't seem to work. So I only did part of the problems. 

The issue is I don't see how RND or DQL help. Or it is just that I failed to make it work.

## Part 1

Note in the original implementation RND can train its exploitation critic 
during exploration phase while random exploration cannot. This is not fair. So 
I change the code to make it fair.

With this change, you would see that RND exploration is not really helpful in
terms of performance. It does provide more uniform density though.


```bash
python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd

python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --unsupervised_exploration --exp_name q1_env1_random

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --exp_name q1_env2_rnd

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --unsupervised_exploration --exp_name q1_env2_random

python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --unsupervised_exploration --exp_name q1_env3_rnd

python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --unsupervised_exploration --exp_name q1_env3_random
```

## Part 2

They are doing equally well. 

```bash
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_dqn --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1
```
