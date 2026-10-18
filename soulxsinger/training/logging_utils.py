"""Per-process terminal capture and concise training reports."""
import logging
import os
from pathlib import Path
import subprocess
import sys
import traceback
from contextlib import contextmanager


@contextmanager
def training_logs(output_dir, rank):
    """Capture file descriptors too, including native libraries and loader workers."""
    directory = Path(output_dir) / 'logs'
    directory.mkdir(parents=True, exist_ok=True)
    suffix = '' if rank == 0 else f'.rank{rank}'
    terminal = directory / f'terminal{suffix}.log'
    report = directory / f'report{suffix}.log'
    logger = logging.getLogger('soulxsinger.train')
    old_level, old_propagate = logger.level, logger.propagate
    handler = logging.FileHandler(report, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    saved = []
    tee = None
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        saved = [os.dup(1), os.dup(2)]
        # An external tee avoids a background Python thread before DataLoader forks.
        tee = subprocess.Popen(['tee', '-a', str(terminal)], stdin=subprocess.PIPE,
                               stdout=saved[0], stderr=saved[1])
        os.dup2(tee.stdin.fileno(), 1)
        os.dup2(tee.stdin.fileno(), 2)
        print(f'Logs: terminal={terminal} report={report}', flush=True)
        logger.info('Run started: rank=%s pid=%s', rank, os.getpid())
        try:
            yield logger
        except BaseException as error:
            logger.error('Run stopped: %s: %s', type(error).__name__, error)
            traceback.print_exc()
            raise SystemExit(130 if isinstance(error, KeyboardInterrupt) else 1)
        else:
            logger.info('Training completed')
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)
        if tee is not None:
            tee.stdin.close()
            try:
                tee.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tee.terminate()
                tee.wait()
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(old_level)
        logger.propagate = old_propagate
