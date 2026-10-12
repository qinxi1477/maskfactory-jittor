"""Native Jittor SDXL UNet assembled from diffusers-compatible blocks."""

from __future__ import annotations

from typing import List

import jittor as jt
from jittor import nn

from masactrl.sd15_jittor import (
    BasicTransformerBlock,
    SafeTensorStore,
    TimestepEmbedding,
    Timesteps,
    UNetDownBlock2D,
    UNetDownsample2D,
    UNetInputStem,
    UNetOutputHead,
    UNetResnetBlock2D,
    UNetUpBlock2D,
    UNetUpsample2D,
    _assign_group_norm,
    _assign_linear,
)


class LinearTransformer(nn.Module):
    def __init__(self, channels: int, heads: int, depth: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.proj_in = nn.Linear(channels, channels)
        self.blocks = [
            BasicTransformerBlock(channels, 2048, heads=heads, dim_head=64)
            for _ in range(depth)
        ]
        self.proj_out = nn.Linear(channels, channels)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_group_norm(self.norm, store, prefix + ".norm")
        _assign_linear(self.proj_in, store, prefix + ".proj_in")
        for index, block in enumerate(self.blocks):
            block.load_from_store(store, f"{prefix}.transformer_blocks.{index}")
        _assign_linear(self.proj_out, store, prefix + ".proj_out")

    def execute(self, value: jt.Var, context: jt.Var) -> jt.Var:
        residual = value
        batch, channels, height, width = value.shape
        hidden = self.norm(value).permute(0, 2, 3, 1).reshape(batch, -1, channels)
        hidden = self.proj_in(hidden)
        for block in self.blocks:
            hidden = block(hidden, context)
            if height * width >= 1024:
                hidden = jt.array(hidden.numpy())
                jt.gc()
        hidden = self.proj_out(hidden)
        hidden = hidden.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return hidden + residual


class CrossDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int, depth: int, downsample: bool):
        super().__init__()
        self.resnets = [
            UNetResnetBlock2D(in_channels, out_channels),
            UNetResnetBlock2D(out_channels, out_channels),
        ]
        self.attentions = [
            LinearTransformer(out_channels, heads, depth),
            LinearTransformer(out_channels, heads, depth),
        ]
        self.downsampler = UNetDownsample2D(out_channels) if downsample else None

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        for index in range(2):
            self.resnets[index].load_from_store(store, f"{prefix}.resnets.{index}")
            self.attentions[index].load_from_store(store, f"{prefix}.attentions.{index}")
        if self.downsampler is not None:
            self.downsampler.load_from_store(store, prefix + ".downsamplers.0")

    def execute(self, value: jt.Var, time: jt.Var, context: jt.Var):
        residuals = []
        for resnet, attention in zip(self.resnets, self.attentions):
            value = attention(resnet(value, time), context)
            residuals.append(value)
        if self.downsampler is not None:
            value = self.downsampler(value)
            residuals.append(value)
        return value, residuals


class CrossUp(nn.Module):
    def __init__(self, inputs: List[int], out_channels: int, heads: int, depth: int, upsample: bool):
        super().__init__()
        self.resnets = [UNetResnetBlock2D(channels, out_channels) for channels in inputs]
        self.attentions = [LinearTransformer(out_channels, heads, depth) for _ in inputs]
        self.upsampler = UNetUpsample2D(out_channels) if upsample else None

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        for index, resnet in enumerate(self.resnets):
            resnet.load_from_store(store, f"{prefix}.resnets.{index}")
            self.attentions[index].load_from_store(store, f"{prefix}.attentions.{index}")
        if self.upsampler is not None:
            self.upsampler.load_from_store(store, prefix + ".upsamplers.0")

    def execute(self, value: jt.Var, time: jt.Var, context: jt.Var, residuals: List[jt.Var]) -> jt.Var:
        for resnet, attention in zip(self.resnets, self.attentions):
            value = jt.concat([value, residuals.pop()], dim=1)
            value = attention(resnet(value, time), context)
        if self.upsampler is not None:
            value = self.upsampler(value)
        return value


class MidBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet0 = UNetResnetBlock2D(1280, 1280)
        self.attention = LinearTransformer(1280, 20, 10)
        self.resnet1 = UNetResnetBlock2D(1280, 1280)

    def load_from_store(self, store: SafeTensorStore) -> None:
        self.resnet0.load_from_store(store, "mid_block.resnets.0")
        self.attention.load_from_store(store, "mid_block.attentions.0")
        self.resnet1.load_from_store(store, "mid_block.resnets.1")

    def execute(self, value: jt.Var, time: jt.Var, context: jt.Var) -> jt.Var:
        value = self.resnet0(value, time)
        value = self.attention(value, context)
        return self.resnet1(value, time)


class SDXLUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = UNetInputStem()
        self.time_proj = Timesteps(320)
        self.time_embed = TimestepEmbedding(320, 1280)
        self.add_proj = Timesteps(256)
        self.add_embed = TimestepEmbedding(2816, 1280)
        self.down0 = UNetDownBlock2D(320, 320, add_downsample=True)
        self.down1 = CrossDown(320, 640, 10, 2, True)
        self.down2 = CrossDown(640, 1280, 20, 10, False)
        self.mid = MidBlock()
        self.up0 = CrossUp([2560, 2560, 1920], 1280, 20, 10, True)
        self.up1 = CrossUp([1920, 1280, 960], 640, 10, 2, True)
        self.up2 = UNetUpBlock2D([960, 640, 640], 320, add_upsample=False)
        self.output = UNetOutputHead()

    def load_from_store(self, store: SafeTensorStore) -> None:
        self.input.load_from_store(store)
        self.time_embed.load_from_store(store, "time_embedding")
        self.add_embed.load_from_store(store, "add_embedding")
        self.down0.load_from_store(store, "down_blocks.0")
        self.down1.load_from_store(store, "down_blocks.1")
        self.down2.load_from_store(store, "down_blocks.2")
        self.mid.load_from_store(store)
        self.up0.load_from_store(store, "up_blocks.0")
        self.up1.load_from_store(store, "up_blocks.1")
        self.up2.load_from_store(store, "up_blocks.2")
        self.output.load_from_store(store)

    def attention_modules(self):
        modules = []

        def add(transformer, place):
            for block in transformer.blocks:
                block.attn1.place = place
                block.attn2.place = place
                modules.extend([block.attn1, block.attn2])

        for transformer in self.down1.attentions + self.down2.attentions:
            add(transformer, "down")
        add(self.mid.attention, "mid")
        for transformer in self.up0.attentions + self.up1.attentions:
            add(transformer, "up")
        return modules

    def condition(self, timestep: jt.Var, pooled: jt.Var, time_ids: jt.Var) -> jt.Var:
        time = self.time_embed(self.time_proj(timestep))
        added = self.add_proj(time_ids.reshape(-1)).reshape(time_ids.shape[0], -1)
        return time + self.add_embed(jt.concat([pooled, added], dim=-1))

    def execute(
        self,
        sample: jt.Var,
        timestep: jt.Var,
        context: jt.Var,
        pooled: jt.Var,
        time_ids: jt.Var,
        trace: dict | None = None,
    ) -> jt.Var:
        time = self.condition(timestep, pooled, time_ids)
        value = self.input(sample)
        residuals = [value]
        value, current = self.down0(value, time, return_residuals=True)
        residuals.extend(current)
        if trace is not None:
            trace["down0"] = value
        value, current = self.down1(value, time, context)
        residuals.extend(current)
        if trace is not None:
            trace["down1"] = value
        value, current = self.down2(value, time, context)
        residuals.extend(current)
        if trace is not None:
            trace["down2"] = value
        value = self.mid(value, time, context)
        if trace is not None:
            trace["mid"] = value
        value = self.up0(value, time, context, residuals)
        if trace is not None:
            trace["up0"] = value
        value = self.up1(value, time, context, residuals)
        if trace is not None:
            trace["up1"] = value
        value = self.up2(value, time, residuals)
        if trace is not None:
            trace["up2"] = value
        return self.output(value)
