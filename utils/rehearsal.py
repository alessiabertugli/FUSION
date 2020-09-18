import random


class ReservoirSampler:
    def __init__(self, windows, buffer_size=5000):
        self.buffer = []
        self.location = 0
        self.buffer_size = buffer_size
        self.window = windows
        self.total_additions = 0

    def add(self, data):
        self.total_additions += 1

        M = len(self.buffer)
        if M < self.buffer_size:
            self.buffer.append(data)
        else:
            i = random.randint(0, min(self.total_additions, self.window))
            if i < self.buffer_size:
                self.buffer[i] = data
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sample_trajectory(self, batch_size):
        initial_index = random.randint(0, len(self.buffer) - batch_size)
        return self.buffer[initial_index: initial_index + batch_size]
