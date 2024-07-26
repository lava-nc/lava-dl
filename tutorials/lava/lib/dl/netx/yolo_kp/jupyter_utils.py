import queue
import multiprocessing
import time
from PIL import Image, ImageDraw

from typing import Optional, Tuple
from IPython.display import display, clear_output, Markdown

class Display:
    def __init__(self,
                 default_frame: Image.Image = None,
                 default_msg: Optional[str]= None,
                 default_size: Tuple[int, int] = (448, 448)) -> None:
        self.data_queue = multiprocessing.Queue()
        self.process = None
        self.frame = default_frame
        if self.frame is None:
            self.frame = Image.new("RGB", default_size, (128, 128, 128))
            draw = ImageDraw.Draw(self.frame)
            msg = 'WAITING FOR DATA'
            W, H = default_size
            draw.text((W // 2, H // 2), msg, fill=(255, 255, 255), anchor='mm')
        self.msg = default_msg if default_msg else ''

    def engine(self) -> None:
        print('Display Message')  # This is necessary
        clear_output(wait=True)
        display(self.frame)
        display(Markdown(self.msg.replace('\n', '<br>')))
        while True:
            num_frames_available = self.data_queue.qsize()
            if num_frames_available > 0:
                for _ in range(num_frames_available):
                    self.frame, self.msg = self.data_queue.get()
                clear_output(wait=True)
                display(self.frame)
                display(Markdown(self.msg.replace('\n', '<br>')))

    def start(self) -> None:
        self.process = multiprocessing.Process(target=self.engine)
        self.process.start()

    def stop(self) -> None:
        while self.data_queue.qsize() > 0:
            time.sleep(0.1)
        self.process.terminate()

    def push(self, frame, msg) -> None:
        self.data_queue.put((frame, msg))
