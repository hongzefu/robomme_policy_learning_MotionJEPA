import os
import json
import subprocess
import numpy as np
import cv2
import imageio
import re
import collections
from pathlib import Path
from typing import Tuple, Optional



TASK_WITH_VIDEO_DEMO = [
    "VideoUnmask", "VideoUnmaskSwap", "VideoPlaceButton", "VideoPlaceOrder",
    "VideoRepick", "MoveCube", "InsertPeg", "PatternLock", "RouteStick"
]

TASK_NAME_LIST=  [      
    "BinFill",
    "StopCube",
    "PickXtimes",
    "SwingXtimes",
    
    "ButtonUnmask",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick"
]

def pack_buffer(image_buffer, state_buffer, exec_start_idx=0):
    image_output = np.stack(image_buffer, axis=0).astype(np.uint8)[:, None]
    state_output = np.stack(state_buffer, axis=0).astype(np.float32)
    return {
        "images": image_output,
        "state": state_output,
        "add_buffer": True,
        "exec_start_idx": exec_start_idx,
    }
    
def check_args(args):
    assert args.obs_horizon == 16



class EpisodeState:
    def __init__(self):
        self.image_buffer = []
        self.wrist_image_buffer = []
        self.state_buffer = []
        self.action_plan = collections.deque()
        self.count = 0
        self.exec_start_idx = 0
        # ── motion 窗统计（0922-binfill-demo-prefix-plan.md C 节）──
        # 这四个字段**绝不能**放进 clear_buffers：那个方法每个 action chunk 调一次，清掉就只剩最后一次推理的数。
        # exec_start_idx_initial 单独存一份，正是因为 clear_buffers 会把 exec_start_idx 归零。
        self.exec_start_idx_initial = 0
        self.motion_infers = 0
        self.motion_k_max = 0
        self.motion_downsample_steps = 0
        self.motion_budget = 0
        self.motion_overflow = ""
        # 最近一次 infer 回传的 motion_k（逐步动作日志用；非 motion 模型恒为 None）
        self.motion_k_last = None

    def add_observation(self, img: np.ndarray, wrist_img: np.ndarray, state: np.ndarray):
        self.image_buffer.append(img.copy())
        self.wrist_image_buffer.append(wrist_img.copy())
        self.state_buffer.append(state.copy())

    def clear_buffers(self):
        self.image_buffer.clear()
        self.wrist_image_buffer.clear()
        self.state_buffer.clear()
        self.exec_start_idx = 0

    def get_current_obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.image_buffer[-1], self.wrist_image_buffer[-1], self.state_buffer[-1]





#: 本客户端进程里已 save_video、尚未回收的 ffmpeg 编码：(proc, 临时名, 最终名, stderr 句柄, 写管道是否断过)。
#: 由 finalize_all() 统一 wait 并改名；每 20 局的客户端进程退出前必须调一次（eval.py::evaluate 的两个退出点）。
_PENDING_ENCODES = []


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class RolloutRecorder:
    """逐帧送进 ffmpeg 子进程流式压缩编码，并逐帧写一行动作日志（1017-test-xhard5-12per T22）。

    与原实现的区别只在落盘方式：原来整局帧攒在内存、局末 ``imageio.mimsave`` 同步写全尺寸视频；
    现在首帧到达时起 ffmpeg（半分辨率、libx264 crf 28、fragmented mp4），每帧经 stdin 管道写入，
    局末 ``save_video`` 只关管道并登记，不等编码结束。帧内容（红框、四条文字条、拼接方式）与原来相同。

    * 临时名以 ``.part`` 结尾（``check_shard.py`` 按 ``*.mp4`` 计数，未改名的不计），回收时 rc=0 才改成最终名；
    * 文字条行数随数值变化会让帧高变化，rawvideo 管道要求定尺寸：首帧定 (H, W)，此后文字条区域按首帧高度
      补零或截断（图像区不动），再断言形状；
    * ffmpeg 提前死亡（BrokenPipeError）只把本局视频记失败，不抛给评估循环；
    * ``steps_path`` 给出时，每次 ``record`` 写一行 jsonl（step、action、state、is_video_demo、motion_k），
      带缓冲、不 flush，``save_video`` / ``discard`` / ``close_steps`` 时关闭。
    """

    def __init__(self, save_dir: str, task_goal: str, fps: int = 30, steps_path=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.task_goal = task_goal
        self.n_frames = 0
        self.failed = False
        self._proc = None
        self._tmp = None
        self._log = None
        self._header_h = None
        self._shape = None
        self._finished = False
        self._steps = None
        if steps_path is not None:
            steps_path = Path(steps_path)
            steps_path.parent.mkdir(parents=True, exist_ok=True)
            self._steps = open(steps_path, "w", encoding="utf-8")

    def record(self, image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray, action: np.ndarray=None,
               is_video_demo: bool=False, motion_k=None):

        concat_image = np.concatenate([image, wrist_image], axis=1)
        if is_video_demo: # add a red border
            concat_image = cv2.rectangle(concat_image, (0, 0), (concat_image.shape[1], concat_image.shape[0]), (255, 0, 0), 10)

        frame_text = "Frame: " + str(self.n_frames)
        frame_text_area = self.add_text_area(frame_text, concat_image.shape)

        goal_text = "Task Goal: " + self.task_goal
        goal_text_area = self.add_text_area(goal_text, concat_image.shape)

        state_text = "State: " + ','.join([f"{i:.4f}" for i in state])
        state_text_area = self.add_text_area(state_text, concat_image.shape)

        action_text = 'Action: ' + ','.join([f"{i:.4f}" for i in action]) if action is not None else "Action:None"
        action_text_area = self.add_text_area(action_text, concat_image.shape)
        # Concatenate text area on top of image
        header = np.concatenate([frame_text_area, goal_text_area, action_text_area, state_text_area], axis=0)
        if self._header_h is None:
            self._header_h = header.shape[0]
        if header.shape[0] < self._header_h:
            pad = np.zeros((self._header_h - header.shape[0],) + header.shape[1:], dtype=header.dtype)
            header = np.concatenate([header, pad], axis=0)
        elif header.shape[0] > self._header_h:
            header = header[:self._header_h]
        frame = np.ascontiguousarray(np.concatenate([header, concat_image], axis=0), dtype=np.uint8)

        if self._steps is not None:
            self._steps.write(json.dumps({
                "step": self.n_frames,
                "action": None if action is None else [float(x) for x in np.asarray(action).reshape(-1)],
                "state": [float(x) for x in np.asarray(state).reshape(-1)],
                "is_video_demo": bool(is_video_demo),
                "motion_k": None if motion_k is None else int(motion_k),
            }) + "\n")
        self._write_frame(frame)
        self.n_frames += 1

    def _write_frame(self, frame: np.ndarray) -> None:
        if self.failed:
            return
        if self._proc is None:
            h, w = frame.shape[:2]
            self._shape = (h, w, 3)
            self._tmp = self.save_dir / f".rec-{os.getpid()}-{id(self):x}-{len(_PENDING_ENCODES)}.mp4.part"
            self._log = open(self.save_dir / "ffmpeg.log", "ab")
            self._proc = subprocess.Popen(
                [_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-",
                 "-vf", "scale=iw/2:-2", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
                 "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", str(self._tmp)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log)
        assert frame.shape == self._shape, (frame.shape, self._shape)
        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as e:
            self.failed = True
            print(f"VIDEO_ENCODE_BROKEN tmp={self._tmp} err={e!r}", flush=True)

    def add_text_area(self, text: str, concat_image_shape: tuple):
        # Calculate text wrapping
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        max_width = concat_image_shape[1] - 20  # Leave 10px margin on each side
        lines = []
        words = text.replace(',', ' ').split()
        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + ' ' + word
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

            if text_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        lines.append(current_line)  # Add the last line

        # Create text area with dynamic height
        line_height = 20
        text_area_height = max(50, len(lines) * line_height + 10)
        text_area = np.zeros((text_area_height, concat_image_shape[1], 3), dtype=np.uint8)

        # Draw each line
        for i, line in enumerate(lines):
            y_position = 15 + i * line_height
            text_area = cv2.putText(text_area, line, (10, y_position), font, font_scale, (255, 255, 255), thickness)

        return text_area

    def close_steps(self) -> None:
        if self._steps is not None:
            self._steps.close()
            self._steps = None

    def _close_stdin(self) -> None:
        try:
            self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            self.failed = True

    def save_video(self, filename: str):
        """关管道并登记，不等编码结束；真正 wait 与改名在 finalize_all()。"""
        self.close_steps()
        if self._finished:
            return
        self._finished = True
        if self._proc is None:
            print(f"VIDEO_ENCODE_EMPTY name={filename}（本局没有帧）", flush=True)
            return
        self._close_stdin()
        _PENDING_ENCODES.append((self._proc, self._tmp, self.save_dir / filename, self._log, self.failed))

    def discard(self):
        """error / unknown 局：关管道、等 ffmpeg 退出、删临时文件；已 save_video 的局不受影响。"""
        self.close_steps()
        if self._finished:
            return
        self._finished = True
        if self._proc is None:
            return
        self._close_stdin()
        self._proc.wait()
        self._log.close()
        try:
            os.remove(self._tmp)
        except FileNotFoundError:
            pass


def finalize_all():
    """回收本进程全部已登记的 ffmpeg：逐个 wait，rc=0 且管道未断则改成最终名，否则删临时文件记失败。

    返回 (回收个数, 失败个数)。
    """
    finalized, failed = 0, 0
    while _PENDING_ENCODES:
        proc, tmp, final, log, broken = _PENDING_ENCODES.pop(0)
        rc = proc.wait()
        log.close()
        finalized += 1
        if rc == 0 and not broken:
            os.replace(tmp, final)
        else:
            failed += 1
            print(f"VIDEO_ENCODE_FAILED name={final.name} rc={rc} broken={broken}", flush=True)
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
    return finalized, failed
