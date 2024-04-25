import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from typing import List, Union, Tuple, Any
from metavision_core.event_io.py_reader import EventDatReader
from metavision_ml.preprocessing import histo_quantized

class VideoRecord(object):
    """
    Helper class for class VideoFrameDataset. This class
    represents a video sample's metadata.

    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: A list with four or more elements where 1) The first
             element is the path to the video sample's frames excluding
             the root_datapath prefix 2) The  second element is the starting frame id of the video
             3) The third element is the inclusive ending frame id of the video
             4) The fourth element is the label index.
             5) any following elements are labels in the case of multi-label classification
    """
    def __init__(self, imgs, path):
        self._imgs = imgs
        self._path = path

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive
    
    @property
    def start_frame(self) -> int:
        return 0

    @property
    def end_frame(self) -> int:
        return len(self._imgs) - 1

class SpikingNTUDataset(torch.utils.data.Dataset):
    r"""
    A highly efficient and adaptable dataset class for videos.
    Instead of loading every frame of a video,
    loads x RGB frames of a video (sparse temporal sampling) and evenly
    chooses those frames from start to end of the video, returning
    a list of x PIL images or ``FRAMES x CHANNELS x HEIGHT x WIDTH``
    tensors where FRAMES=x if the ``ImglistToTensor()``
    transform is used.

    More specifically, the frame range [START_FRAME, END_FRAME] is divided into NUM_SEGMENTS
    segments and FRAMES_PER_SEGMENT consecutive frames are taken from each segment.

    Note:
        A demonstration of using this class can be seen
        in ``demo.py``
        https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch

    Note:
        This dataset broadly corresponds to the frame sampling technique
        introduced in ``Temporal Segment Networks`` at ECCV2016
        https://arxiv.org/abs/1608.00859.


    Note:
        This class relies on receiving video data in a structure where
        inside a ``ROOT_DATA`` folder, each video lies in its own folder,
        where each video folder contains the frames of the video as
        individual files with a naming convention such as
        img_001.jpg ... img_059.jpg.
        For enumeration and annotations, this class expects to receive
        the path to a .txt file where each video sample has a row with four
        (or more in the case of multi-label, see README on Github)
        space separated values:
        ``VIDEO_FOLDER_PATH     START_FRAME      END_FRAME      LABEL_INDEX``.
        ``VIDEO_FOLDER_PATH`` is expected to be the path of a video folder
        excluding the ``ROOT_DATA`` prefix. For example, ``ROOT_DATA`` might
        be ``home\data\datasetxyz\videos\``, inside of which a ``VIDEO_FOLDER_PATH``
        might be ``jumping\0052\`` or ``sample1\`` or ``00053\``.

    Args:
        annotationfile_path: The .txt annotation file containing
                             one row per video sample as described above.
        num_segments: The number of segments the video should
                      be divided into to sample frames from.
        frames_per_segment: The number of frames that should
                            be loaded per segment. For each segment's
                            frame-range, a random start index or the
                            center is chosen, from which frames_per_segment
                            consecutive frames are loaded.
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders as described above.
        transform: Transform pipeline that receives a list of PIL images/frames.
        test_mode: If True, frames are taken from the center of each
                   segment, instead of a random location in each segment.

    """
    def __init__(self,
                 annotationfile_path: str,
                 num_segments: int = 30,
                 frames_per_segment: int = 1,
                 transform = None,
                 test_mode: bool = False,
                 classify_labels: Union[list, str] = "all"):
        super().__init__()

        self.annotation_file_path = annotationfile_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.num_frames = self.num_segments * self.frames_per_segment
        self.transform = transform
        self.test_mode = test_mode
        self.classify_labels = classify_labels
        if self.classify_labels == 'all':
            self.label_map = {i+1: i for i in range(120)}
        else:
            self.label_map = {v: i+1 for i, v in enumerate(self.classify_labels)}

        self.fns = []
        self.label_list = []
        with open(self.annotation_file_path, "r") as f:
            for entry in f:
                fn, label = entry.split(" ")
                self.fns.append(fn)
                self.label_list.append(self.label_map.get(int(label), 0))

        # TODO offer those as parameters 
        self.dt = 10**6 / 150
        self.tau = self.dt * 5
    
    @property
    def num_classes(self):
        if self.classify_labels == 'all':
            return len(self.label_map)
        else:
            # When we specify a subset of labels, the 0 class implicitly includes the remaining classes
            return len(self.label_map) + 1

    def balance_by_random_drop(self, force_min=None):
        """
        Balances the dataset by dropping random samples.
        """

        class_counts = np.bincount(self.label_list)
        if force_min:
            min_class_count = force_min
        else:
            min_class_count = np.min(class_counts)

        num_samples_to_drop = [np.max([0, cc - min_class_count]) for cc in class_counts]

        for class_id, to_drop in enumerate(num_samples_to_drop):
            ids = np.where(np.array(self.label_list) == class_id)[0]
            ids_to_drop = np.random.choice(ids, to_drop, replace=False)
            self.label_list = [label for id, label in enumerate(self.label_list) if id not in ids_to_drop]
            self.fns = [fn for id, fn in enumerate(self.fns) if id not in ids_to_drop]

    def _get_start_indices(self, record: VideoRecord) -> 'np.ndarray[int]':
        """
        For each segment, choose a start index from where frames
        are to be loaded from.

        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """
        num_frames_needed = self.num_segments * self.frames_per_segment
        if record.num_frames < num_frames_needed:
            # return all frames and repeat the first one
            start_indices = (num_frames_needed - record.num_frames) * [record.start_frame] + list(range(record.start_frame, record.end_frame + 1))
            return start_indices
        # choose start indices that are perfectly evenly spread across the video frames.
        if self.test_mode:
            distance_between_indices = (record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)

            start_indices = np.array([int(distance_between_indices / 2.0 + distance_between_indices * x)
                                      for x in range(self.num_segments)])
        # randomly sample start indices that are approximately evenly spread across the video frames.
        else:
            max_valid_start_index = (record.num_frames - self.frames_per_segment + 1) // self.num_segments

            start_indices = np.multiply(list(range(self.num_segments)), max_valid_start_index) + \
                      np.random.randint(max_valid_start_index, size=self.num_segments)

        return start_indices + record.start_frame

    def __getitem__(self, idx: int):
        """
        For video with id idx, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
        frames from evenly chosen locations across the video.

        Args:
            idx: Video sample index.

        """

        fn = self.fns[idx]
        label = self.label_list[idx]

        record_dat = EventDatReader(fn)
        if record_dat.event_count() == 0:
            print("EMPTY", fn)
        height, width = record_dat.get_size()
        tbins=1
        img = np.zeros([3, width, height])
        imgs = [img.T]

        while record_dat.current_time / 10**6 < record_dat.duration_s:
            events = record_dat.load_delta_t(self.dt)
            volume = np.zeros((1, 2, height, width), dtype=np.uint8)

            histo_quantized(events, volume, self.dt)
            img[:] += volume[0, 0].astype(bool).T + volume[0, 1].astype(bool).T
            imgs.append(img.copy().T)
            img *= 0#np.exp(-(self.dt / self.tau))

            # histo_quantized(events, volume, self.dt)
            # img[:] += volume[0, 0].astype(np.int32).T - volume[0, 1].astype(np.int32).T
            # imgs.append(img.copy().T)
            # img *= np.exp(-(self.dt / self.tau))

        
        record: VideoRecord = VideoRecord(imgs, fn)

        frame_start_indices: 'np.ndarray[int]' = self._get_start_indices(record)

        frames = self._get(record, frame_start_indices).float()

        # clear memory
        del record._imgs
        del record
        
        return frames, label

    def _get(self, record: VideoRecord, frame_start_indices: 'np.ndarray[int]') -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple['torch.Tensor[num_frames, channels, height, width]', Union[int, List[int]]],
        Tuple[Any, Union[int, List[int]]],
        ]:
        """
        Loads the frames of a video at the corresponding
        indices.

        Args:
            record: VideoRecord denoting a video sample.
            frame_start_indices: Indices from which to load consecutive frames from.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """

        images = []
        # from each start_index, load self.frames_per_segment
        # consecutive frames
        for start_index in frame_start_indices:
            frame_index = int(start_index)

            # load self.frames_per_segment consecutive frames
            for _ in range(self.frames_per_segment):
                # deal with short movies by zero padding in front 
                if len(record._imgs) < self.num_frames:
                    frame_index -= self.num_frames - len(record._imgs) 
                if frame_index < 0:
                    image = np.zeros_like(record._imgs[0])
                else:
                    image = record._imgs[frame_index]
                if self.transform:
                    image = self.transform(image)
                else:
                    image = torch.from_numpy(image)
                images.append(image)

                if frame_index < record.end_frame:
                    frame_index += 1

        images = torch.stack(images)

        return images

    def __len__(self):
        return len(self.fns)


class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``VideoFrameDataset``.
    """
    @staticmethod
    def forward(img_list: List[Image.Image]) -> 'torch.Tensor[NUM_IMAGES, CHANNELS, HEIGHT, WIDTH]':
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.

        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])
