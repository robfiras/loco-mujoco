from copy import deepcopy


class GoalDirectionVelocity:

    def __init__(self):
        self._direction = None
        self._velocity = None

    def __call__(self):
        return self.get_goal()

    def get_goal(self):
        assert self._direction is not None
        assert self._velocity is not None
        return deepcopy(self._direction), deepcopy(self._velocity)

    def set_goal(self, direction, velocity):
        self._direction = direction
        self._velocity = velocity

    def get_direction(self):
        assert self._direction is not None
        return deepcopy(self._direction)

    def get_velocity(self):
        assert self._velocity is not None
        return deepcopy(self._velocity)
