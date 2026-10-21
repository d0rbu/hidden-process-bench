from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)
