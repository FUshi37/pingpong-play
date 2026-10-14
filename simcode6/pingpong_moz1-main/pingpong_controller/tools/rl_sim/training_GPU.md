mujoco mjx GPU训练：
python train_juggle_mjx_curriculum.py \
  --n-envs 1024 \
  --n-steps 256 \
  --minibatch-size 16384 \
  --update-epochs 4 \
  --advance-mode converged \
  --save-dir /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_curriculum_vx \
  --wandb \
  --wandb-project pingpong-mjx \
  --wandb-name mjx-curriculum-vx

resume训练：
python train_juggle_mjx_curriculum.py \
  --resume-from /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_curriculum_v5/10_stage3a_smooth_hardware_limited_action.pkl \
  --resume-start-stage stage3b_light_camera_constraint \
  --n-envs 1024 \
  --n-steps 256 \
  --minibatch-size 16384 \
  --update-epochs 4 \
  --advance-mode converged \
  --save-dir /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_curriculum_vx \
  --wandb \
  --wandb-project pingpong-mjx \
  --wandb-name mjx-curriculum-vx

stage4g resume:
python train_juggle_mjx_curriculum.py \
  --resume-from /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_curriculum_v7/18_stage4g_strong_contact_dr.pkl \
  --resume-start-stage stage4g_strong_contact_dr \
  --max-stages 18 \
  --advance-mode fixed \
  --stage-steps 1000000000 \
  --n-envs 1024 \
  --n-steps 256 \
  --minibatch-size 16384 \
  --update-epochs 4 \
  --save-every-updates 10 \
  --save-dir /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_stage4g_continue_v1 \
  --wandb \
  --wandb-project pingpong-mjx \
  --wandb-name mjx-stage4g-continue-v1

更严格的curriclum gate:
python train_juggle_mjx_curriculum.py \
  --n-envs 1024 \
  --n-steps 256 \
  --minibatch-size 16384 \
  --update-epochs 4 \
  --learning-rate 1e-4 \
  --clip-range 0.1 \
  --advance-mode converged \
  --curriculum-gate-preset v7_strict \
  --advance-validation-mode block \
  --advance-eval-n-envs 128 \
  --save-dir /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_curriculum_v8 \
  --wandb \
  --wandb-project pingpong-mjx \
  --wandb-name mjx-curriculum-v8

validate:
python validate_juggle_mjx_ppo.py --checkpoint /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_stage4g_continue_v1/18_stage4g_strong_contact_dr.pkl --episodes 3 --n-envs 1 --deterministic --render --realtime

录制视频:
MUJOCO_GL=egl python validate_juggle_mjx_ppo.py   --checkpoint /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_stage4g_continue_v1/18_stage4g_strong_contact_dr.pkl   --episodes 1   --n-envs 1   --deterministic   --video-out /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/demo_stage4g.gif   --video-fps 30   --video-width 1280   --video-height 720