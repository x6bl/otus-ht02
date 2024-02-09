import numpy as np
import random


class ReplayBuffer:

    def __init__(
        self,
        max_size: int = 50000,
        obs_n: int = 4
    ) -> None:
        self.max_size = max_size
        self.obs_n = obs_n
        self.cnt = 0
        self.st_buf = np.zeros((self.max_size, self.obs_n))
        self.ac_buf = np.zeros((self.max_size, 1), dtype=np.int32)
        self.rw_buf = np.zeros((self.max_size))
        self.ns_buf = np.zeros((self.max_size, self.obs_n))
        self.dn_buf = np.zeros((self.max_size), dtype=np.bool_)

    def __len__(self) -> int:
        return self.cnt

    def push(
        self,
        st: np.ndarray,
        ac: int,
        rw: float,
        ns: np.ndarray,
        dn: bool
    ) -> None:
        if self.cnt < self.max_size:
            self.st_buf[self.cnt,:] = st
            self.ac_buf[self.cnt,0] = ac
            self.rw_buf[self.cnt] = rw
            self.ns_buf[self.cnt,:] = ns
            self.dn_buf[self.cnt] = dn
            self.cnt += 1
        else:
            self.st_buf[:-1] = self.st_buf[1:]
            self.ac_buf[:-1] = self.ac_buf[1:]
            self.rw_buf[:-1] = self.rw_buf[1:]
            self.ns_buf[:-1] = self.ns_buf[1:]
            self.dn_buf[:-1] = self.dn_buf[1:]
            self.st_buf[-1,:] = st
            self.ac_buf[-1,0] = ac
            self.rw_buf[-1] = rw
            self.ns_buf[-1,:] = ns
            self.dn_buf[-1] = dn

    def sample(
        self,
        batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if self.cnt < 2 * batch_size:
            return None
        i = np.array(random.sample(range(self.cnt), batch_size))
        st_batch = self.st_buf[i]
        ac_batch = self.ac_buf[i]
        rw_batch = self.rw_buf[i]
        ns_batch = self.ns_buf[i]
        dn_batch = self.dn_buf[i]
        return (st_batch, ac_batch, rw_batch, ns_batch, dn_batch)

