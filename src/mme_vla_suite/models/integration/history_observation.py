from flax import struct

from openpi.models.model import ArrayT
from openpi.models.model import Observation as _Observation
from openpi.models.model import preprocess_observation as _preprocess_observation
import openpi.shared.array_typing as at


#  b: batch size
#  t: time step
#  v: view
#  h: height
#  w: width
#  c: channel
#  l: seq len


@at.typecheck
@struct.dataclass
class HistAugObservation(_Observation):
    # for perceptual memory
    static_image_emb: at.Float[at.Array, "b l1 d1"] | None = None
    static_mask: at.Bool[at.Array, "b l1"] | None = None
    static_pos_emb: at.Float[at.Array, "b l1 d2"] | None = None
    static_state_emb: at.Float[at.Array, "b l1 d3"] | None = None
    # motion memory（motion-memory-plan.md 2.2）：追加在四个 static_* 之后，mem_order 排在三个 motion_* 之后；
    # l4 = motion.budget（96），d4 = motion.dim（768），d5 = motion.pos_dim（256）；
    # mem_order 用 at.Int（jaxtyping 的 Float 白名单不含 int32）与新维名 l5 = budget + motion.budget（608），
    # 不得复用 l1/l4（@at.typecheck 把全部字段塞进同一个 memo，同名维必须同值）
    motion_emb: at.Float[at.Array, "b l4 d4"] | None = None
    motion_pos: at.Float[at.Array, "b l4 d5"] | None = None
    motion_mask: at.Bool[at.Array, "b l4"] | None = None
    mem_order: at.Int[at.Array, "b l5"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "HistAugObservation":
        parent_obs = super().from_dict(data)
        return cls(
            # Base observation fields
            images=parent_obs.images,
            image_masks=parent_obs.image_masks,
            state=parent_obs.state,
            tokenized_prompt=parent_obs.tokenized_prompt,
            tokenized_prompt_mask=parent_obs.tokenized_prompt_mask,
            token_ar_mask=parent_obs.token_ar_mask,
            token_loss_mask=parent_obs.token_loss_mask,
            # MMEVLA fields
            static_image_emb=data.get("static_image_emb", None),
            static_mask=data.get("static_mask", None),
            static_pos_emb=data.get("static_pos_emb", None),
            static_state_emb=data.get("static_state_emb", None),
            motion_emb=data.get("motion_emb", None),
            motion_pos=data.get("motion_pos", None),
            motion_mask=data.get("motion_mask", None),
            mem_order=data.get("mem_order", None),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        result = super().to_dict()
        result["static_image_emb"] = self.static_image_emb
        result["static_mask"] = self.static_mask
        result["static_pos_emb"] = self.static_pos_emb
        result["static_state_emb"] = self.static_state_emb
        result["motion_emb"] = self.motion_emb
        result["motion_pos"] = self.motion_pos
        result["motion_mask"] = self.motion_mask
        result["mem_order"] = self.mem_order
        return result

    def to_base_obs(self) -> _Observation:
        return _Observation(
            images=self.images,
            image_masks=self.image_masks,
            state=self.state,
            tokenized_prompt=self.tokenized_prompt,
            tokenized_prompt_mask=self.tokenized_prompt_mask,
            token_ar_mask=self.token_ar_mask,
            token_loss_mask=self.token_loss_mask,
        )

    @classmethod
    def from_base_obs(
        cls,
        base_obs: _Observation,
        static_image_emb: at.Float[ArrayT, "*b l d1"] | None = None,
        static_mask: at.Bool[ArrayT, "*b l"] | None = None,
        static_pos_emb: at.Float[ArrayT, "*b l d2"] | None = None,
        static_state_emb: at.Float[ArrayT, "*b l d3"] | None = None,
        # 本签名内 l 已被 512 占住，motion 侧另用一套维名；jaxtyped 装饰 dataclass 只包 __init__，方法注解不做运行时检查
        motion_emb: at.Float[ArrayT, "*b m d4"] | None = None,
        motion_pos: at.Float[ArrayT, "*b m d5"] | None = None,
        motion_mask: at.Bool[ArrayT, "*b m"] | None = None,
        mem_order: at.Int[ArrayT, "*b lm"] | None = None,
    ) -> "HistAugObservation":
        return HistAugObservation(
            images=base_obs.images,
            image_masks=base_obs.image_masks,
            state=base_obs.state,
            tokenized_prompt=base_obs.tokenized_prompt,
            tokenized_prompt_mask=base_obs.tokenized_prompt_mask,
            token_ar_mask=base_obs.token_ar_mask,
            token_loss_mask=base_obs.token_loss_mask,
            static_image_emb=static_image_emb,
            static_mask=static_mask,
            static_pos_emb=static_pos_emb,
            static_state_emb=static_state_emb,
            motion_emb=motion_emb,
            motion_pos=motion_pos,
            motion_mask=motion_mask,
            mem_order=mem_order,
        )


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: HistAugObservation,
    *args,
    **kwargs,
) -> HistAugObservation:
    base_obs: _Observation = _preprocess_observation(
        rng,
        observation.to_base_obs(),
        *args,
        **kwargs,
    )
    return HistAugObservation.from_base_obs(
        base_obs,
        static_image_emb=observation.static_image_emb,
        static_mask=observation.static_mask,
        static_pos_emb=observation.static_pos_emb,
        static_state_emb=observation.static_state_emb,
        motion_emb=observation.motion_emb,
        motion_pos=observation.motion_pos,
        motion_mask=observation.motion_mask,
        mem_order=observation.mem_order,
    )
