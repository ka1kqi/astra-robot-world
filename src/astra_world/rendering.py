"""Render actual MuJoCo data on its owner thread for the browser viewport."""

from io import BytesIO
import mujoco
from PIL import Image


class WorldRenderer:
    def __init__(self, width=960, height=640):
        self.width, self.height = width, height
        self._renderer = None
        self._model = None
        self._camera = mujoco.MjvCamera()

    def render(self, world):
        if self._model is not world.model:
            self.close()
            self._renderer = mujoco.Renderer(
                world.model, height=self.height, width=self.width
            )
            self._model = world.model
            self._camera.lookat[:] = world.model.stat.center
            self._camera.distance = max(1.2, world.model.stat.extent * 1.6)
            if getattr(world.spec, "robot", "panda") == "panda":
                self._camera.lookat[:] = [0.32, 0, 0.32]
                self._camera.distance = 1.65
            self._camera.azimuth = 145
            self._camera.elevation = -35
        self._renderer.update_scene(world.data, camera=self._camera)
        pixels = self._renderer.render()
        output = BytesIO()
        Image.fromarray(pixels).save(output, format="JPEG", quality=82)
        return output.getvalue()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = self._model = None
