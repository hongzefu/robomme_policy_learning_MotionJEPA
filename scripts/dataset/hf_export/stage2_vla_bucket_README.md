# stage2 1024 维 VLA checkpoint

本桶对应唯一训练 run `mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k`。
帧路为8帧×8×8，motion每窗4×256按时间槽拼成1×1024；窗口33帧、步长32、预算96。
MotionJEPA来源为 `stage2-full144-1025-a/checkpoint_epoch_105.pt#model`，使用EMA。

保存的16步为5000至75000（每5000步）及79999；60000用于后续测试。
每个step直接保留Orbax目录结构。根文件包括实际解析的history配置及SHA、完整motion来源、
原子ready清单、G0/G2/G3与恢复receipt及SOURCE_PIN记录，以及encoder依赖清单。
`online_restore_inputs.json`与对应NPZ携带真实demo、连续exec、state、wrist与prompt恢复素材，
其SHA同时绑定在JSON与ready中，异机验收不依赖原机器隐藏的pkl或H5。
`encoder_dependencies/` 内有ep105与config的实体副本，恢复时按交付根相对路径定位，
不依赖 `source_path` 所记的原机器路径。

`manifests/root.json`绑定完整根快照和encoder依赖；`manifests/step-60000.json`同时绑定
该step与根快照；最终的 `delivery_manifest.json` 绑定完整发布集合，每项有SHA256和字节数。
发布清单不包含自身；父清单或下载方另行记录其SHA。下载相应清单列出的每一个文件后，
逐文件校验，再通过仓库的 `check_stage2_checkpoint.py --mode delivery` 严格恢复与真实在线推理。
VAE权重来源身份与版本锁保存在 `stage2_compat.online_vae` 中，
按 `vae_id` 的HF模型 `vae` 子目录准备权重并核 `vae_state_sha256`。
恢复环境显式设置 `MMEVLA_V1_STORE`，只使用其 `cache/hf` 缓存；不依赖隐式用户缓存。
恢复素材的tokenizer记录固定取得URI与SHA256，按该身份准备
`OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model`，显式设置 `OPENPI_DATA_HOME` 并核SHA。
VAE与tokenizer身份可跨机器验证，不能用未知版本替代。

训练期入口等待原子ready，先本地stage、来源合同与公开信息扫描，通过 `ROOT_LOCAL_SMOKE=PASS`
后才触网。最终驱动须先以 `SMOKE=1` 对同一全量清单预检，再运行正式上传。

`.done`只表示传输与匿名SHA回读成功。只有独立恢复检查输出 `CKPT_60K_READY=PASS`
才表示60000可交测试；训练期间上传完毕不能代替此项验证。最终全部16步及根文件
匿名回读通过才输出 `CKPT_UPLOAD=PASS`。本README不宣称尚未发生的训练或恢复验收成功。

公开信息披露：为保持原始SHA，配置、来源证据、依赖清单与训练日志保留原机器的
绝对路径、源码仓库引用、内部主机标识；它们只作来源记录，不作为跨机器身份。
凭据不在允许清单内，公开前检测到疑似凭据会停止上传。
