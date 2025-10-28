import torch
import math
import comfy.utils
import logging


class CONDRegular:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return self.__class__(cond)

    def process_cond(self, batch_size, **kwargs):
        return self._copy_with(comfy.utils.repeat_to_batch_size(self.cond, batch_size))

    def can_concat(self, other):
        if self.cond.shape != other.cond.shape:
            return False
        if self.cond.device != other.cond.device:
            logging.warning("WARNING: conds not on same device, skipping concat.")
            return False
        return True

    def concat(self, others):
        conds = [self.cond]
        for x in others:
            conds.append(x.cond)
        return torch.cat(conds)

    def size(self):
        return list(self.cond.size())


class CONDNoiseShape(CONDRegular):
    def process_cond(self, batch_size, area, **kwargs):
        data = self.cond
        if area is not None:
            dims = len(area) // 2
            for i in range(dims):
                data = data.narrow(i + 2, area[i + dims], area[i])

        return self._copy_with(comfy.utils.repeat_to_batch_size(data, batch_size))


class CONDCrossAttn(CONDRegular):
    def can_concat(self, other):
        s1 = self.cond.shape
        s2 = other.cond.shape
        if s1 != s2:
            if s1[0] != s2[0] or s1[2] != s2[2]:  # these 2 cases should not happen
                return False

            mult_min = math.lcm(s1[1], s2[1])
            diff = mult_min // min(s1[1], s2[1])
            if (
                diff > 4
            ):  # arbitrary limit on the padding because it's probably going to impact performance negatively if it's too much
                return False
        if self.cond.device != other.cond.device:
            logging.warning("WARNING: conds not on same device: skipping concat.")
            return False
        return True

    def concat(self, others):
        conds = [self.cond]
        lengths = [self.cond.shape[1]]
        for x in others:
            c = x.cond
            lengths.append(c.shape[1])
            conds.append(c)

        # Compute the LCM of all sequence lengths
        crossattn_max_len = lengths[0]
        for length in lengths[1:]:
            crossattn_max_len = math.lcm(crossattn_max_len, length)

        # Preallocate output tensor for efficiency if all shapes and types are compatible
        # Otherwise, repeat and collect as in original code
        out = []
        for c, length in zip(conds, lengths):
            if length < crossattn_max_len:
                factor = crossattn_max_len // length
                # Avoid unnecessary repeat if factor == 1
                c = c if factor == 1 else c.repeat(1, factor, 1)
            out.append(c)

        # torch.cat can be slow if the list is large; for 3D tensors sharing dim 0, dim 1 it's best to use torch.cat at once
        return torch.cat(out)


class CONDConstant(CONDRegular):
    def __init__(self, cond):
        self.cond = cond

    def process_cond(self, batch_size, **kwargs):
        return self._copy_with(self.cond)

    def can_concat(self, other):
        if self.cond != other.cond:
            return False
        return True

    def concat(self, others):
        return self.cond

    def size(self):
        return [1]


class CONDList(CONDRegular):
    def __init__(self, cond):
        self.cond = cond

    def process_cond(self, batch_size, **kwargs):
        out = []
        for c in self.cond:
            out.append(comfy.utils.repeat_to_batch_size(c, batch_size))

        return self._copy_with(out)

    def can_concat(self, other):
        if len(self.cond) != len(other.cond):
            return False
        for i in range(len(self.cond)):
            if self.cond[i].shape != other.cond[i].shape:
                return False

        return True

    def concat(self, others):
        out = []
        for i in range(len(self.cond)):
            o = [self.cond[i]]
            for x in others:
                o.append(x.cond[i])
            out.append(torch.cat(o))

        return out

    def size(self):  # hackish implementation to make the mem estimation work
        o = 0
        c = 1
        for c in self.cond:
            size = c.size()
            o += math.prod(size)
            if len(size) > 1:
                c = size[1]

        return [1, c, o // c]
