import torch.utils.data as data
from PIL import Image
import os
import numpy as np
from numpy.random import randint


__all__ = ["RubiksDataset"]


class RubiksDataset(data.Dataset):
    def __init__(
        self,
        root_path,
        list_file,
        num_segments=3,
        new_length=1,
        image_tmpl="img_{:05d}.jpg",
        transform=None,
        random_shift=True,
        test_mode=False,
        remove_missing=False,
        dense_sample=False,
        all_sample=False,
        twice_sample=False,
        only_even_indices=True,
        logger=None,
    ):
        self.verbose = logger is not None
        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample
        self.all_sample = all_sample
        self.twice_sample = twice_sample
        self.only_even_indices = only_even_indices

        if self.verbose:
            if self.dense_sample:
                self.print("=> Using dense sample for the dataset...")
            if self.all_sample:
                print("=> Using all sample for the dataset...")
            if self.twice_sample:
                print("=> Using twice sample for the dataset...")

        self._parse_list()

    def print(self, *x):
        if self.verbose:
            print(*x)

    def _load_image(self, directory, idx):
        try:
            return [
                Image.open(
                    os.path.join(self.root_path, directory, self.image_tmpl.format(idx))
                ).convert("RGB")
            ]
        except Exception:
            print(
                "error loading image:",
                os.path.join(self.root_path, directory, self.image_tmpl.format(idx)),
            )
            return [
                Image.open(
                    os.path.join(self.root_path, directory, self.image_tmpl.format(2))
                ).convert("RGB")
            ]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(" ") for x in open(self.list_file)]
        if not self.test_mode or self.remove_missing:
            self.print("=> Removing missing...")
            tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == "{:06d}-{}_{:05d}.jpg":
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        self.print("video number:%d" % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            if self.only_even_indices:
                sample_pos = max(1, 1 + record.num_frames // 2 - 32)
                t_stride = 32 // self.num_segments
                start_idx = (
                    0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
                )
                offsets = [
                    (idx * t_stride + start_idx) % (record.num_frames // 2)
                    for idx in range(self.num_segments)
                ]
                return (np.array(offsets) + 1) * 2
            else:
                sample_pos = max(1, 1 + record.num_frames - 64)
                t_stride = 64 // self.num_segments
                start_idx = (
                    0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
                )
                offsets = [
                    (idx * t_stride + start_idx) % record.num_frames
                    for idx in range(self.num_segments)
                ]
                return np.array(offsets) + 1
        elif self.all_sample:
            sample_pos = max(1, 1 + record.num_frames - self.num_segments)
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [
                (idx + start_idx) % record.num_frames
                for idx in range(self.num_segments)
            ]
            return np.array(offsets) + 1
        else:  # normal sample
            if not self.only_even_indices:
                average_duration = (
                    record.num_frames - self.new_length + 1
                ) // self.num_segments
                if average_duration > 0:
                    offsets = np.multiply(
                        list(range(self.num_segments)), average_duration
                    ) + randint(average_duration, size=self.num_segments)
                elif record.num_frames > self.num_segments:
                    offsets = np.sort(
                        randint(
                            record.num_frames - self.new_length + 1,
                            size=self.num_segments,
                        )
                    )
                else:
                    offsets = np.zeros((self.num_segments,))
                return offsets + 1
            else:
                if self.dense_sample:
                    print("video too short...")
                average_duration = (
                    record.num_frames // 2 - self.new_length + 1
                ) // self.num_segments
                if average_duration > 0:
                    offsets = np.multiply(
                        list(range(self.num_segments)), average_duration
                    ) + randint(average_duration, size=self.num_segments)
                elif record.num_frames // 2 > self.num_segments:
                    offsets = np.sort(
                        randint(
                            record.num_frames // 2 - self.new_length + 1,
                            size=self.num_segments,
                        )
                    )
                else:
                    offsets = np.zeros((self.num_segments,))
                return (offsets + 1) * 2

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            if self.only_even_indices:
                sample_pos = max(1, 1 + record.num_frames // 2 - 32)
                t_stride = 32 // self.num_segments
                start_idx = (
                    0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
                )
                offsets = [
                    (idx * t_stride + start_idx) % (record.num_frames // 2)
                    for idx in range(self.num_segments)
                ]
                return (np.array(offsets) + 1) * 2
            else:
                sample_pos = max(1, 1 + record.num_frames - 64)
                t_stride = 64 // self.num_segments
                start_idx = (
                    0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
                )
                offsets = [
                    (idx * t_stride + start_idx) % record.num_frames
                    for idx in range(self.num_segments)
                ]
                return np.array(offsets) + 1
        elif self.all_sample:
            sample_pos = max(1, 1 + record.num_frames - self.num_segments)
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [
                (idx + start_idx) % record.num_frames
                for idx in range(self.num_segments)
            ]
            return np.array(offsets) + 1
        else:  # strided
            if not self.only_even_indices:
                if record.num_frames > self.num_segments + self.new_length - 1:
                    tick = (record.num_frames - self.new_length + 1) / float(
                        self.num_segments
                    )
                    offsets = np.array(
                        [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
                    )
                else:
                    offsets = np.zeros((self.num_segments,))
                return offsets + 1
            else:
                if record.num_frames // 2 > self.num_segments + self.new_length - 1:
                    tick = (record.num_frames // 2 - self.new_length + 1) / float(
                        self.num_segments
                    )
                    offsets = np.array(
                        [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
                    )
                else:
                    offsets = np.zeros((self.num_segments,))
                return (offsets + 1) * 2

    def _get_test_indices(self, record):
        if self.dense_sample:
            if self.only_even_indices:
                sample_pos = max(1, 1 + record.num_frames // 2 - 32)
                t_stride = 32 // self.num_segments
                start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
                offsets = []
                for start_idx in start_list.tolist():
                    offsets += [
                        (idx * t_stride + start_idx) % (record.num_frames // 2)
                        for idx in range(self.num_segments)
                    ]
                return (np.array(offsets) + 1) * 2
            else:
                sample_pos = max(1, 1 + record.num_frames - 64)
                t_stride = 64 // self.num_segments
                start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
                offsets = []
                for start_idx in start_list.tolist():
                    offsets += [
                        (idx * t_stride + start_idx) % record.num_frames
                        for idx in range(self.num_segments)
                    ]
                return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array(
                [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
                + [int(tick * x) for x in range(self.num_segments)]
            )

            return offsets + 1
        elif self.all_sample:
            offsets = np.arange(record.num_frames)
            return np.array(offsets) + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array(
                [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
            )

            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.image_tmpl == "{:06d}-{}_{:05d}.jpg":
            file_name = self.image_tmpl.format(
                int(record.path), "x", 2 if self.only_even_indices else 1
            )
            full_path = os.path.join(
                self.root_path, "{:06d}".format(int(record.path)), file_name
            )
        else:
            file_name = self.image_tmpl.format(2 if self.only_even_indices else 1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        while not os.path.exists(full_path):
            print(
                "################## Not Found:",
                os.path.join(self.root_path, record.path, file_name),
            )
            raise ValueError("not found")
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == "{:06d}-{}_{:05d}.jpg":
                file_name = self.image_tmpl.format(
                    int(record.path), "x", 2 if self.only_even_indices else 1
                )
                full_path = os.path.join(
                    self.root_path, "{:06d}".format(int(record.path)), file_name
                )
            else:
                file_name = self.image_tmpl.format(2 if self.only_even_indices else 1)
                full_path = os.path.join(self.root_path, record.path, file_name)

        if not self.test_mode:
            segment_indices = (
                self._sample_indices(record)
                if self.random_shift
                else self._get_val_indices(record)
            )
        else:
            segment_indices = self._get_test_indices(record)
        ret = self.get(record, segment_indices)
        return ret

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoRecordDuration(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

    @property
    def duration(self):  # in sec
        return float(self._data[3])
