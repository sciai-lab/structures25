"""Provides `CallbackTiming` to configure when to call a callback.

The `CallbackTiming` class is used to configure when a callback should be called.
This is useful for expensive callbacks that should not be called every step, such as image logging.
"""


class CallbackTiming:
    """Base class for callback timings."""

    def call_now(self, global_step: int) -> bool:
        """Returns True if the callback should be called now.

        Args:
            global_step: The current global step.

        Returns:
            True if the callback should be called now.
        """
        raise NotImplementedError


class EveryNSteps(CallbackTiming):
    """Calls the callback every `n_steps` steps."""

    def __init__(self, n_steps: int) -> None:
        """Initializes the EveryNSteps object.

        Args:
            n_steps: The number of steps between calls.
        """
        self.n_steps = n_steps
        self.next_log = 0

    def call_now(self, global_step: int) -> bool:
        """Returns True if the callback should be called now, which is the case if the global step
        is larger than `self.next_log`, the next scheduled call.

        .. Note::
            If `call_now` is not called every step, the callback might not be called at all the scheduled steps at even
            intervals. This happens e.g. when using this at validation time.

        Args:
            global_step: The current global step.

        Returns:
            True if the callback should be called now.
        """
        if global_step >= self.next_log:
            self.next_log = global_step + self.n_steps
            return True
        else:
            return False


class EveryIncreasingInterval(CallbackTiming):
    """Callback timing with an exponentially increasing interval between calls."""

    def __init__(
        self, initial_log_step=0, initial_interval: int = 10, slow_down_factor: float = 1.1
    ) -> None:
        """Calls the callback every `log_interval` steps, increasing the interval by
        `slow_down_factor` every time. This results in exponentially increasing intervals between
        calls, and log files that grow logarithmically.

        Args:
            initial_log_step: The first step at which the callback should be called.
            initial_interval: The initial number of steps between calls.
            slow_down_factor: The factor by which the interval is increased after each call.
        """
        self.log_next = initial_log_step
        self.log_interval = initial_interval
        self.slow_down_factor = slow_down_factor

    def call_now(self, global_step: int) -> bool:
        """Returns True if the callback should be called now, which is the case if the global step
        is larger than `self.next_log`, the next scheduled call.

        .. Note::
            If `call_now` is not called every step, the callback might not be called at all the scheduled steps at
            exponentially increasing intervals. This happens e.g. when using this at validation time.

        Args:
            global_step: The current global step.

        Returns:
            True if the callback should be called now.
        """
        if global_step >= self.log_next:
            self.log_next = global_step + self.log_interval
            self.log_interval = self.log_interval * self.slow_down_factor
            return True
        else:
            return False
