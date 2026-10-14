import csv
import os
from collections import defaultdict


class FinetuneLogger:
    def __init__(self, output_dir, print_every=100, csv_every=None):
        self.output_dir = str(output_dir)
        self.print_every = print_every
        self.csv_every = csv_every

        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "metrics.csv")

        self._csv_sum = defaultdict(float)
        self._csv_count = 0
        self._epoch_sum = defaultdict(float)
        self._epoch_count = 0
        self._prt_sum = defaultdict(float)
        self._prt_min = defaultdict(lambda: float("inf"))
        self._prt_max = defaultdict(lambda: float("-inf"))
        self._prt_count = 0

        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(["epoch", "batch", "loss", "l1", "lpips"])

    def log_batch(self, epoch, batch_idx, loss, l1, lpips):
        for key, val in [("loss", loss), ("l1", l1), ("lpips", lpips)]:
            self._csv_sum[key] += val
            self._epoch_sum[key] += val
        self._csv_count += 1
        self._epoch_count += 1

        write_every = self.csv_every if self.csv_every is not None else 1
        if self._csv_count >= write_every:
            self._flush_csv_window(epoch, batch_idx)

        for key, val in [("loss", loss), ("l1", l1), ("lpips", lpips)]:
            self._prt_sum[key] += val
            if val < self._prt_min[key]:
                self._prt_min[key] = val
            if val > self._prt_max[key]:
                self._prt_max[key] = val
        self._prt_count += 1

        if self._prt_count >= self.print_every:
            self._flush_print(epoch, batch_idx)

    def log_epoch(self, epoch):
        csv_partial = self._csv_count > 0
        if csv_partial:
            self._flush_csv_window(epoch, self._csv_count)

        e_loss = self._epoch_sum["loss"] / max(self._epoch_count, 1)
        e_l1 = self._epoch_sum["l1"] / max(self._epoch_count, 1)
        e_lpips = self._epoch_sum["lpips"] / max(self._epoch_count, 1)
        e_total = self._epoch_count

        self._writer.writerow([epoch, -1, f"{e_loss:.6f}", f"{e_l1:.6f}", f"{e_lpips:.6f}"])
        self._csv_file.flush()
        self._epoch_sum = defaultdict(float)
        self._epoch_count = 0
        self._reset_csv()

        prt_count = self._prt_count
        if prt_count > 0:
            self._flush_print(epoch, "end")
        if e_total > 0:
            print(f"Epoch {epoch}: avg_loss={e_loss:.6f}")
        else:
            print(f"Epoch {epoch}: avg_loss=N/A")
        self._reset_print()

    def _flush_csv_window(self, epoch, batch_idx):
        avg_loss = self._csv_sum["loss"] / self._csv_count
        avg_l1 = self._csv_sum["l1"] / self._csv_count
        avg_lpips = self._csv_sum["lpips"] / self._csv_count
        self._writer.writerow([epoch, batch_idx, f"{avg_loss:.6f}", f"{avg_l1:.6f}", f"{avg_lpips:.6f}"])
        self._csv_file.flush()
        self._reset_csv()

    def _flush_print(self, epoch, batch):
        avg = {k: self._prt_sum[k] / self._prt_count for k in ("loss", "l1", "lpips")}
        print(
            f"  [E{epoch} B{batch}] "
            f"loss(avg={avg['loss']:.6f} min={self._prt_min['loss']:.6f} max={self._prt_max['loss']:.6f}) "
            f"l1(avg={avg['l1']:.6f}) lpips(avg={avg['lpips']:.6f})"
        )
        self._reset_print()

    def _reset_csv(self):
        self._csv_sum = defaultdict(float)
        self._csv_count = 0

    def _reset_print(self):
        self._prt_sum = defaultdict(float)
        self._prt_min = defaultdict(lambda: float("inf"))
        self._prt_max = defaultdict(lambda: float("-inf"))
        self._prt_count = 0

    def close(self):
        self._csv_file.close()
