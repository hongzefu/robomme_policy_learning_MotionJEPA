# 阶段II与阶段III的pool4三解码器checkpoint

本桶包含两个独立源run，各自保留目录前缀、config、逐轮metrics及文件清单：

- `stage2-full144-1025-a/`：阶段III，epoch5至105每5轮一份，共21份；另有 `motion_ablation/` 原始结果。
- `stage2-scale-20261025-d/`：阶段II，epoch4、epoch8、final三份。

阶段III在第106轮停止，无 `checkpoint_final.pt`，本存档不表示144轮正常完成。
两个run的同名文件不会覆盖。`manifests/<run>.json`分别绑定真实文件SHA256、字节数
与逐份模型schema检查结果；`delivery_manifest.json`绑定最终发布集合。

架构为 `wan-latent-stage2-blockwise-pool4-three-decoder`，每份包含encoder、RGB、XYZ与state decoder。
`weight_type=ema`，`model`为EMA，实际存在的 `model_live` 为live权重，
`optimizer`为原训练状态，均原样保留，不假称仅有EMA。
每份checkpoint通过config、fingerprint摘要与 `fingerprint.data_metadata` 重建真实模型，
严格核完整185个模型状态项（其中encoder65项）的键、shape、dtype及有限值；
用 `model` 恢复，不把两份权重键混用。

正式上传前，必须先以 `SMOKE=1` 对同一完整清单做本地预检；预检不访问凭据或网络。
下载后先按发布清单全量核SHA与字节数，再使用锁定源码的严格checkpoint加载接口。
只有全量匿名回读与源文件未变核验都通过，驱动才输出 `MJ_CKPT_UPLOAD=PASS`。

公开信息披露：config、metrics、消融来源记录含原机器路径和MotionJEPA源码仓库引用，
为保持SHA原样发布；这些位置不代替可移植的相对路径清单。疑似凭据始终禁止发布。
