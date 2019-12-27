"""save checkpoint information"""
import os
import shutil
import torch
import pdb

from  ..utils import logs
from  ..utils import file
from  ..utils import timeutil
import json,time,heapq

class MinHeap(object):
    def __init__(self, key):
        self.key = key
        self.heap = []

    def push(self, item):
        decorated = self.key(item), item
        heapq.heappush(self.heap, decorated)

    def pop(self):
        key, item = heapq.heappop(self.heap)
        return item

    def pushpop(self, item):
        decorated = self.key(item), item
        rkey, ritem = heapq.heappushpop(self.heap, decorated)
        return ritem

    def __len__(self):
        return len(self.heap)

    def __getitem__(self, index):
        return self.heap[index][1]


class Checkpointer(object):
    def __init__(self, save_dir, keep_best=5, keep_last=5):
        self.save_dir = save_dir
        self.keep_best = keep_best
        self.keep_last = keep_last
        self.checkpoints = MinHeap(lambda x: x[0])

        self.ckpt_pattern = os.path.join(self.save_dir, "checkpoint_{}.pt")
        self.glob_pattern = os.path.join(self.save_dir, "checkpoint_*.pt")
        self.meta_path = os.path.join(self.save_dir, "checkpoint_meta.txt")

        self._read_meta()

    def _read_meta(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf8") as f:
                for line in f:
                    accuracy, ckpt = line.split()
                    accuracy = float(accuracy)
                    if len(self.checkpoints) < self.keep_best:
                        self.checkpoints.push((accuracy, ckpt))
                    else:
                        self.checkpoints.pushpop((accuracy, ckpt))

    def _write_meta(self):
        with open(self.meta_path, "w", encoding="utf8") as f:
            for acc, ckpt in self.checkpoints:
                f.write(f"{acc}\t{ckpt}\n")
                f.flush()

    def _cleanup(self, ckpts_to_keep):
        ckpts = glob.glob(self.glob_pattern)
        for ckpt in ckpts:
            if ckpt not in ckpts_to_keep:
                os.remove(ckpt)
                print(f"checkpoint {ckpt} is removed")

    def save(self, last_epoch, data, accuracy=None):
        start = time.time()
        ckpt_name = self.ckpt_pattern.format(last_epoch)
        torch.save(data, ckpt_name)
        print(
            f"checkpoint saved to {ckpt_name} (used {time.time() - start:.3f}s)"
        )

        if (
            accuracy is not None
            and self.keep_best is not None
            and self.keep_best > 0
        ):
            if len(self.checkpoints) < self.keep_best:
                self.checkpoints.push((accuracy, ckpt_name))
            else:
                self.checkpoints.pushpop((accuracy, ckpt_name))
            best_checkpoints = [x[1] for x in self.checkpoints]
        else:
            best_checkpoints = []

        if self.keep_last is not None and self.keep_last > 0:
            last_checkpoints = [
                self.ckpt_pattern.format(last_epoch - x)
                for x in range(self.keep_last)
            ]
        else:
            last_checkpoints = []

        if best_checkpoints or last_checkpoints:
            checkpoints = set(best_checkpoints + last_checkpoints)
            self._cleanup(checkpoints)

        if self.checkpoints:
            self._write_meta()

    def load(self, epoch):
        if epoch >= 0:
            ckpt_name = self.ckpt_pattern.format(epoch)
            if os.path.exists(ckpt_name):
                state_dict = torch.load(ckpt_name, "cpu")
                print(f"checkpoint {ckpt_name} loaded")
                return (
                    state_dict["net"],
                    state_dict["optimizer"],
                    state_dict["last_epoch"],
                )
            print(f"checkpoint {ckpt_name} does not exist")
        return None, None, None