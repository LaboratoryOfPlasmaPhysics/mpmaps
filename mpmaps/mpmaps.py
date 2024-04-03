import numpy as np


class MPMap:
    def __init__(self, clock, cone, tilt, **kwargs):
        self._clock = clock
        self._cone = cone
        self._tilt = tilt
        self._bimf = kwargs.get("bimf", 5)
        self._nsw = kwargs.get("nsw", 5)
        self.ny = 400  # hard coded for now just to declare values
        self.nz = 400
        self.values = np.zeros((self.ny, self.nz))

    def _rotate_msh(self, dclock):
        pass

    def _read_bmsh(self, cone, tilt):
        return self.values  # just to get an ndarray, should read the grid instead

    def _read_bmsp(self, cone, tilt):
        return self.values  # just to get an ndarray, should read the grid instead

    def set_clock(self, clock):
        self._rotate_msh(self._clock - clock)
        self._update()
        self._clock = clock


class ShearMap(MPMap):
    def __init__(self, clock, cone, tilt, **kwargs):
        super(ShearMap, self).__init__(clock, cone, tilt, **kwargs)
        self._bmsh = self._read_bmsh(cone, tilt)
        self._bmsp = self._read_bmsp(cone, tilt)
        self.values = self._update()

    def _update(self):
        print("updating shear")
        # return compute the angle from self._bmsh and self._bmsp
        return self.values  # just to return something for now


class RateMap(MPMap):
    def __init__(self, clock, cone, tilt, **kwargs):
        super(RateMap, self).__init__(clock, cone, tilt, **kwargs)
        self._bmsh = self._read_bmsh(cone, tilt)
        self._bmsp = self._read_bmsh(cone, tilt)
        self._nmsh = self._read_nmsh()
        self._nmsp = self._read_nmsp()
        self.values = self._update()

    def _read_nmsh(self):
        return self.values  # just to get an ndarray, should read the grid instead

    def _read_nmsp(self):
        return self.values  # just to get an ndarray, should read the grid instead

    def _update(self):
        print("updating rate")
        return self.values


class JMap(MPMap):
    def __init__(self, clock, cone, tilt, **kwargs):
        super(JMap, self).__init__(clock, cone, tilt, **kwargs)
