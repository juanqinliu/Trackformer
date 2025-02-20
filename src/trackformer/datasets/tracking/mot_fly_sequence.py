# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT_FLY sequence dataset.
"""
import configparser
import csv
import os
import os.path as osp
from argparse import Namespace
from typing import Optional, Tuple, List

import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset
from .mot17_sequence import MOT17Sequence

from ..coco import make_coco_transforms
from ..transforms import Compose


class MOTFLYSequence(Dataset):
    """Multiple Object Tracking (MOT_FLY) Dataset.

    This dataloader is designed so that it can handle only one sequence,
    if more have to be handled one should inherit from this class.
    """
    data_folder = 'MOT_FLY_V0.4/MOT_FLY_V0.3'

    def __init__(self, root_dir: str = '/home/ljq/UAV-Tracking/Dataset', 
                 seq_name: Optional[str] = None,
                 vis_threshold: float = 0.0, 
                 img_transform: Namespace = None) -> None:
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
            img_transform: Image transformations
        """
        super().__init__()

        self._seq_name = seq_name
        self._vis_threshold = vis_threshold
        self._data_dir = os.path.join(root_dir, self.data_folder)

        self.transforms = Compose(make_coco_transforms('val', img_transform, overflow_boxes=True))

        self.data = []
        self.no_gt = True
        if seq_name is not None:
            # Try both train and test directories
            train_path = os.path.join(self._data_dir, 'train', seq_name)
            test_path = os.path.join(self._data_dir, 'test', seq_name)
            
            if os.path.exists(train_path):
                self._data_dir = os.path.join(self._data_dir, 'train')
            elif os.path.exists(test_path):
                self._data_dir = os.path.join(self._data_dir, 'test')
            else:
                raise AssertionError(
                    f'Sequence not found: {seq_name}\n'
                    f'Tried paths:\n'
                    f'  {train_path}\n'
                    f'  {test_path}'
                )

            self.data = self._sequence()
            self.no_gt = not osp.exists(self.get_gt_file_path())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        width_orig, height_orig = img.size

        img, _ = self.transforms(img)
        width, height = img.size(2), img.size(1)

        sample = {}
        sample['img'] = img
        sample['dets'] = torch.tensor(np.array([det[:4] for det in data['dets']]))
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']
        sample['orig_size'] = torch.as_tensor([int(height_orig), int(width_orig)])
        sample['size'] = torch.as_tensor([int(height), int(width)])

        return sample

    def _sequence(self) -> List[dict]:
        """Get sequence information."""
        # Public detections
        dets = {i: [] for i in range(1, self.seq_length + 1)}
        det_file = self.get_det_file_path()

        if osp.exists(det_file):
            with open(det_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    x1 = float(row[2]) - 1
                    y1 = float(row[3]) - 1
                    x2 = x1 + float(row[4]) - 1
                    y2 = y1 + float(row[5]) - 1
                    score = float(row[6])
                    bbox = np.array([x1, y1, x2, y2, score], dtype=np.float32)
                    dets[int(float(row[0]))].append(bbox)

        img_dir = osp.join(self.get_seq_path(), 'img1')
        boxes, visibility = self.get_track_boxes_and_visbility()

        total = [
            {'gt': boxes[i],
             'im_path': osp.join(img_dir, f"{i:06d}.jpg"),
             'vis': visibility[i],
             'dets': dets[i]}
            for i in range(1, self.seq_length + 1)]

        return total

    def get_track_boxes_and_visbility(self) -> Tuple[dict, dict]:
        """Load ground truth boxes and their visibility."""
        boxes = {}
        visibility = {}

        for i in range(1, self.seq_length + 1):
            boxes[i] = {}
            visibility[i] = {}

        gt_file = self.get_gt_file_path()
        if not osp.exists(gt_file):
            return boxes, visibility

        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                if float(row[8]) >= self._vis_threshold:
                    x1 = int(float(row[2])) - 1
                    y1 = int(float(row[3])) - 1
                    x2 = x1 + int(float(row[4])) - 1
                    y2 = y1 + int(float(row[5])) - 1
                    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

                    frame_id = int(row[0])
                    track_id = int(row[1])

                    boxes[frame_id][track_id] = bbox
                    visibility[frame_id][track_id] = float(row[8])

        return boxes, visibility

    def get_seq_path(self) -> str:
        """Return directory path of sequence."""
        return osp.join(self._data_dir, self._seq_name)

    def get_gt_file_path(self) -> str:
        """Return ground truth file of sequence."""
        return osp.join(self.get_seq_path(), 'gt', 'gt.txt')

    def get_det_file_path(self) -> str:
        """Return public detections file of sequence."""
        return osp.join(self.get_seq_path(), 'det', 'det.txt')

    @property
    def seq_length(self) -> int:
        """Return sequence length, i.e, number of frames."""
        img_dir = osp.join(self.get_seq_path(), 'img1')
        return len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    def __str__(self) -> str:
        return self._seq_name

    @property
    def results_file_name(self) -> str:
        """Generate file name of results file."""
        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"
        return f"{self._seq_name}.txt"

    def write_results(self, results: dict, output_dir: str) -> None:
        """Write the tracks in the format for MOT_FLY submission.

        results: dictionary with 1 dictionary for every track with
                {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        result_file_path = osp.join(output_dir, self.results_file_name)

        with open(result_file_path, "w") as r_file:
            writer = csv.writer(r_file, delimiter=',')

            for i, track in results.items():
                for frame, data in track.items():
                    x1 = data['bbox'][0]
                    y1 = data['bbox'][1]
                    x2 = data['bbox'][2]
                    y2 = data['bbox'][3]

                    writer.writerow([
                        frame + 1,
                        i + 1,
                        x1 + 1,
                        y1 + 1,
                        x2 - x1 + 1,
                        y2 - y1 + 1,
                        -1, -1, -1, -1])

    def load_results(self, results_dir: str) -> dict:
        """Load tracking results from file.

        Args:
            results_dir (str): Path to results directory
            
        Returns:
            dict: Tracking results
        """
        results = {}
        if results_dir is None:
            return results

        file_path = osp.join(results_dir, self.results_file_name)

        if not os.path.isfile(file_path):
            return results

        with open(file_path, "r") as file:
            csv_reader = csv.reader(file, delimiter=',')

            for row in csv_reader:
                frame_id, track_id = int(row[0]) - 1, int(row[1]) - 1

                if track_id not in results:
                    results[track_id] = {}

                x1 = float(row[2]) - 1
                y1 = float(row[3]) - 1
                x2 = float(row[4]) - 1 + x1
                y2 = float(row[5]) - 1 + y1

                results[track_id][frame_id] = {}
                results[track_id][frame_id]['bbox'] = [x1, y1, x2, y2]
                results[track_id][frame_id]['score'] = 1.0

        return results