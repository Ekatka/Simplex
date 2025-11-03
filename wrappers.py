import gymnasium as gym


class EarlyStopWrapper(gym.Wrapper):
    def __init__(self, env, max_degenerate_streak=100, window=200, improve_tol=1e-12):
        super().__init__(env)
        self.max_degenerate_streak = int(max_degenerate_streak)
        self.window = int(window)
        self.improve_tol = float(improve_tol)
        self._last_obj = None
        self._no_improve_steps = 0

    def reset(self, **kwargs):
        self._last_obj = None
        self._no_improve_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)

        obj = float(info.get("objective", self.env.T[-1, -1]))
        if self._last_obj is None:
            self._last_obj = obj
        else:
            delta = self._last_obj - obj
            if delta > self.improve_tol:
                self._no_improve_steps = 0
            else:
                self._no_improve_steps += 1
            self._last_obj = obj

        if info.get("degenerate_streak", 0) >= self.max_degenerate_streak:
            truncated = True

        if self._no_improve_steps >= self.window:
            truncated = True

        return obs, rew, done, truncated, info


class MacroStrategyWrapper(gym.Wrapper):
    def __init__(self, env, macro_len: int = 10):
        super().__init__(env)
        self.macro_len = macro_len
        self._remaining = 0
        self._current_strategy = None

    def reset(self, **kwargs):
        self._remaining = 0
        self._current_strategy = None
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._remaining == 0:
            self._current_strategy = int(action)
            self._remaining = self.macro_len
        obs, reward, terminated, truncated, info = self.env.step(self._current_strategy)
        self._remaining -= 1
        info = dict(info)
        info["macro_strategy"] = self._current_strategy
        info["macro_remaining"] = self._remaining
        return obs, reward, terminated, truncated, info


